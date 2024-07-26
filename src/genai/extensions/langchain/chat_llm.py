"""Wrapper around IBM GENAI APIs for use in Langchain"""

import logging
from pathlib import Path

from pydantic import ConfigDict
from pydantic.v1 import validator

from genai import Client
from genai._types import EnumLike
from genai._utils.general import to_model_optional
from genai.extensions._common.utils import (
    _prepare_chat_generation_request,
    create_generation_info_from_response,
)
from genai.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ModerationParameters,
    SystemMessage,
    TextGenerationParameters,
    TrimMethod,
)

import json
import uuid
import re
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Iterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    ToolCall,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.base import RunnableMap
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.tools import BaseTool

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage as LCAIMessage
    from langchain_core.messages import BaseMessage as LCBaseMessage
    from langchain_core.messages import ChatMessage as LCChatMessage
    from langchain_core.messages import HumanMessage as LCHumanMessage
    from langchain_core.messages import SystemMessage as LCSystemMessage
    from langchain_core.messages import get_buffer_string
    from langchain_core.outputs import ChatGeneration, ChatResult

    from genai.extensions.langchain.utils import (
        CustomAIMessageChunk,
        CustomChatGenerationChunk,
        create_llm_output,
        dump_optional_model,
        load_config,
        update_token_usage_stream,
    )
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")  # noqa: B904

__all__ = ["LangChainChatInterface"]

logger = logging.getLogger(__name__)

Message = Union[LCBaseMessage, BaseMessage]
Messages = Union[list[LCBaseMessage], list[Message]]


def _convert_message_to_genai(message: Message) -> BaseMessage:
    def convert_message_content(content: Any) -> str:
        if not isinstance(content, str):
            raise TypeError(
                f"Cannot convert non-string message content. Got {content} of type {type(content)}, expected string."
            )

        return content

    if isinstance(message, BaseMessage):
        return message
    elif isinstance(message, LCChatMessage) or isinstance(message, LCHumanMessage):
        return HumanMessage(content=convert_message_content(message.content))
    elif isinstance(message, LCAIMessage):
        return AIMessage(content=convert_message_content(message.content))
    elif isinstance(message, LCSystemMessage):
        return SystemMessage(content=convert_message_content(message.content))
    else:
        raise ValueError(f"Got unknown message type '{message}'")


def _convert_messages_to_genai(messages: Messages) -> list[BaseMessage]:
    return [_convert_message_to_genai(msg) for msg in messages]


class LangChainChatInterface(BaseChatModel):
    """
    Class representing the LangChainChatInterface for interacting with the LangChain chat API.

    Example::

        from genai import Client, Credentials
        from genai.extensions.langchain import LangChainChatInterface
        from langchain_core.messages import HumanMessage, SystemMessage
        from genai.schema import TextGenerationParameters

        client = Client(credentials=Credentials.from_env())
        llm = LangChainChatInterface(
            client=client,
            model_id="meta-llama/llama-3-70b-instruct",
            parameters=TextGenerationParameters(
                max_new_tokens=250,
            )
        )

        response = chat_model.generate(messages=[HumanMessage(content="Hello world!")])
        print(response)

    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    client: Client
    model_id: str
    prompt_id: Optional[str] = None
    parameters: Optional[TextGenerationParameters] = None
    moderations: Optional[ModerationParameters] = None
    parent_id: Optional[str] = None
    prompt_template_id: Optional[str] = None
    trim_method: Optional[EnumLike[TrimMethod]] = None
    use_conversation_parameters: Optional[bool] = None
    conversation_id: Optional[str] = None
    streaming: Optional[bool] = None

    @validator("parameters", "moderations", pre=True, always=True)
    @classmethod
    def validate_data_models(cls, value, values, config, field):
        return to_model_optional(value, Model=field.type_, copy=False)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"client": "CLIENT"}

    @classmethod
    def load_from_file(cls, file: Union[str, Path], *, client: Client):
        config = load_config(file)
        return cls(**config, client=client)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "parameters": dump_optional_model(self.parameters),
            "moderations": dump_optional_model(self.moderations),
            "parent_id": self.parent_id,
            "prompt_template_id": self.prompt_template_id,
            "trim_method": self.trim_method,
            "use_conversation_parameters": self.use_conversation_parameters,
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        return "ibmgenai_chat_llm"

    def _prepare_request(self, **kwargs):
        updated = {k: kwargs.pop(k, v) for k, v in self._identifying_params.items()}
        return _prepare_chat_generation_request(**kwargs, **updated)

    def _stream(
        self,
        messages: Messages,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[CustomChatGenerationChunk]:
        for response in self.client.text.chat.create_stream(
            **self._prepare_request(messages=_convert_messages_to_genai(messages), stop=stop, **kwargs)
        ):
            if not response:
                continue

            def send_chunk(*, text: str = "", generation_info: dict):
                logger.info("Chunk received: {}".format(text))
                chunk = CustomChatGenerationChunk(
                    message=CustomAIMessageChunk(content=text, generation_info=generation_info),
                    generation_info=generation_info,
                )
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(token=text, chunk=chunk, response=response)  # noqa: B023
                    # Function definition does not bind loop variable `response`: linter is probably just confused here

            if response.moderations:
                generation_info = create_generation_info_from_response(response, result=response.moderations)
                yield from send_chunk(generation_info=generation_info)

            for result in response.results or []:
                generation_info = create_generation_info_from_response(response, result=result)
                yield from send_chunk(text=result.generated_text or "", generation_info=generation_info)

    def _generate(
        self,
        messages: Messages,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        def handle_stream():
            final_generation: Optional[CustomChatGenerationChunk] = None
            for result in self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                if final_generation:
                    token_usage = result.generation_info.pop("token_usage")
                    final_generation += result
                    update_token_usage_stream(
                        target=final_generation.generation_info["token_usage"],
                        source=token_usage,
                    )
                else:
                    final_generation = result

            assert final_generation and final_generation.generation_info
            return {
                "text": final_generation.text,
                "generation_info": final_generation.generation_info.copy(),
            }

        def handle_non_stream():
            response = self.client.text.chat.create(
                **self._prepare_request(messages=_convert_messages_to_genai(messages), stop=stop, **kwargs),
            )

            assert response.results
            result = response.results[0]

            return {
                "text": result.generated_text or "",
                "generation_info": create_generation_info_from_response(response, result=result),
            }

        result = handle_stream() if self.streaming else handle_non_stream()
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=LCAIMessage(content=result["text"]),
                    generation_info=result["generation_info"].copy(),
                )
            ],
            llm_output=create_llm_output(
                model=result["generation_info"].get("model_id", self.model_id or ""),
                token_usages=[result["generation_info"]["token_usage"]],
            ),
        )

    def get_num_tokens(self, text: str) -> int:
        response = list(self.client.text.tokenization.create(model_id=self.model_id, input=[text]))[0]
        return response.results[0].token_count

    def get_num_tokens_from_messages(self, messages: list[LCBaseMessage]) -> int:
        return sum(
            sum(result.token_count for result in response.results)
            for response in self.client.text.tokenization.create(
                model_id=self.model_id, input=[get_buffer_string([message]) for message in messages]
            )
        )

    def _combine_llm_outputs(self, llm_outputs: list[Optional[dict]]) -> dict:
        token_usages: list[Optional[dict]] = []
        model = ""

        for output in llm_outputs:
            if output:
                model = model or output.get("meta", {}).get("model_id")
                token_usages.append(output.get("token_usage"))

        return create_llm_output(model=model or self.model_id, token_usages=token_usages)

    def get_token_ids(self, text: str) -> list[int]:
        raise NotImplementedError("API does not support returning token ids.")


DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}

"""  # noqa: E501

DEFAULT_RESPONSE_FUNCTION = {
    "name": "__conversational_response",
    "description": (
        "Respond conversationally if no other tools should be called for a given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Conversational response to the user.",
            },
        },
        "required": ["response"],
    },
}

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and (
        issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


def _is_pydantic_object(obj: Any) -> bool:
    return isinstance(obj, BaseModel)


def ibm_json_parser(response):
    json_pattern = re.compile(r'{.*}', re.DOTALL)
    match = json_pattern.search(response)
    
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    else:
        raise ValueError("No JSON found in the response.")


def convert_to_ibm_tool(tool: Any) -> Dict:
    """Convert a tool to an ibm tool."""
    description = None
    if _is_pydantic_class(tool):
        schema = tool.construct().schema()
        name = schema["title"]
    elif _is_pydantic_object(tool):
        schema = tool.get_input_schema().schema()
        name = tool.get_name()
        description = tool.description
    elif isinstance(tool, dict) and "name" in tool and "parameters" in tool:
        return tool.copy()
    else:
        raise ValueError(
            f"""Cannot convert {tool} to an ibm tool. 
            {tool} needs to be a Pydantic class, model, or a dict."""
        )
    definition = {"name": name, "parameters": schema}
    if description:
        definition["description"] = description

    return definition

class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def parse_response(message: BaseMessage) -> str:
    """Extract `function_call` from `AIMessage`."""
    if isinstance(message, AIMessage):
        kwargs = message.additional_kwargs
        tool_calls = message.tool_calls
        if len(tool_calls) > 0:
            tool_call = tool_calls[-1]
            args = tool_call.get("args")
            return json.dumps(args)
        elif "function_call" in kwargs:
            if "arguments" in kwargs["function_call"]:
                return kwargs["function_call"]["arguments"]
            raise ValueError(
                f"`arguments` missing from `function_call` within AIMessage: {message}"
            )
        else:
            raise ValueError("`tool_calls` missing from AIMessage: {message}")
    raise ValueError(f"`message` is not an instance of `AIMessage`: {message}")


class LangChainChatFunctions(LangChainChatInterface):
    """Function chat model that uses IBM API."""

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(functions=tools, **kwargs)

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model o    utput will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.


        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        llm = self.bind_tools(tools=[schema])
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticOutputParser(
                pydantic_object=schema
            )
        else:
            output_parser = JsonOutputParser()

        parser_chain = RunnableLambda(parse_response) | output_parser
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_chain, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | parser_chain

    def _convert_messages_to_ibm_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ibm_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage) or isinstance(message, ToolMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for ibm.")

            content = ""
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ibm_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return ibm_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]
        functions = [convert_to_ibm_tool(fn) for fn in functions]
        functions.append(DEFAULT_RESPONSE_FUNCTION)
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        formatted_messages = [system_message] + messages
        while (isinstance(formatted_messages[0], SystemMessage) and isinstance(formatted_messages[1], SystemMessage)):
            combined_message = SystemMessage(formatted_messages[0].content + formatted_messages[1].content)
            formatted_messages = [combined_message] + formatted_messages[2:]
        response_message = super()._generate(
            formatted_messages, stop=stop, run_manager=run_manager, **kwargs
        )
        chat_generation_content = response_message.generations[0].text
        if not isinstance(chat_generation_content, str):
            raise ValueError("ibmFunctions does not support non-string output.")
        try:
            parsed_chat_result = ibm_json_parser(chat_generation_content)
            print(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"""'{self.model_id}' did not respond with valid JSON. 
                Please try again. 
                Response: {chat_generation_content}"""
            )
        called_tool_name = (
            parsed_chat_result["tool"] if "tool" in parsed_chat_result else None
        )
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )
        if (
            called_tool is None
            or called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]
        ):
            if (
                "tool_input" in parsed_chat_result
                and "response" in parsed_chat_result["tool_input"]
            ):
                response = parsed_chat_result["tool_input"]["response"]
            elif "response" in parsed_chat_result:
                response = parsed_chat_result["response"]
            else:
                print('Multiple Tools Detected')
                tools_to_call = []
                for item in parsed_chat_result:
                    if item['type'] == 'json':
                        content = item['content']
                        called_tool_name = content['tool'] if 'tool' in content else None
                        called_tool = next((fn for fn in functions if fn["name"] == called_tool_name), None)
                        called_tool_arguments = content["tool_input"] if "tool_input" in content else {}
                        tool_call_info = ToolCall(name=called_tool_name, args=called_tool_arguments if called_tool_arguments else {}, id=f"call_{str(uuid.uuid4()).replace('-', '')}")
                        tools_to_call.append(tool_call_info)
                # TODO: make content actually a string with text outside tool calls/
                response_message_with_functions = AIMessage(
                    content="",
                    tool_calls=tools_to_call
                )
                return ChatResult(
                    generations=[ChatGeneration(message=response_message_with_functions)]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=response,
                        )
                    )
                ]
            )

        called_tool_arguments = (
            parsed_chat_result["tool_input"]
            if "tool_input" in parsed_chat_result
            else {}
        )

        response_message_with_functions = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name=called_tool_name,
                    args=called_tool_arguments if called_tool_arguments else {},
                    id=f"call_{str(uuid.uuid4()).replace('-', '')}",
                )
            ],
        )

        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]
        elif not functions:
            functions.append(DEFAULT_RESPONSE_FUNCTION)
        if _is_pydantic_class(functions[0]):
            functions = [convert_to_ibm_tool(fn) for fn in functions]
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        response_message = await super()._agenerate(
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        chat_generation_content = response_message.generations[0].text
        if not isinstance(chat_generation_content, str):
            raise ValueError("ibmFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"""'{self.model_id}' did not respond with valid JSON. 
                Please try again. 
                Response: {chat_generation_content}"""
            )
        called_tool_name = parsed_chat_result["tool"]
        called_tool_arguments = parsed_chat_result["tool_input"]
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )
        if called_tool is None:
            raise ValueError(
                f"Failed to parse a function call from {self.model_id} output: "
                f"{chat_generation_content}"
            )
        if called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=called_tool_arguments["response"],
                        )
                    )
                ]
            )

        response_message_with_functions = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": called_tool_name,
                    "arguments": json.dumps(called_tool_arguments)
                    if called_tool_arguments
                    else "",
                },
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    @property
    def _llm_type(self) -> str:
        return "ibm_functions"