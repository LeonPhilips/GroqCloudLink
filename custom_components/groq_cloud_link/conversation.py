"""Conversation support for Groq Cloud."""

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Literal

import groq
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.chat_completion_message_tool_call_param import Function
from groq.types.shared_params.function_definition import FunctionDefinition
from groq.types.shared_params.function_parameters import FunctionParameters
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    AssistantContentDeltaDict,
    ChatLog,
    Content,
    SystemContent,
    ToolResultContent,
    UserContent,
    async_get_chat_log,
)
from homeassistant.components.conversation.const import ConversationEntityFeature
from homeassistant.components.conversation.entity import ConversationEntity
from homeassistant.components.conversation.models import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, MATCH_ALL

from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.intent import IntentResponse
from homeassistant.util.ulid import ulid_now
from voluptuous_openapi import convert

from .const import (
    CONF_MODEL_IDENTIFIER,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DOMAIN,
    SUBENTRY_MODEL_PARAMS,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _fix_invalid_arguments(value: str) -> Any:
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def domain_fixup(
    obj: Any,
) -> Any:
    """Replace the domain schema type to a string instead of an array."""
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = domain_fixup(obj[i])
        return obj

    if not isinstance(obj, dict):
        return obj

    if "domain" in obj:
        target = obj["domain"]
        if "type" in target and target["type"] == "array":
            target["type"] = "string"
            del target["items"]

    for key in obj.keys():  # noqa: SIM118
        obj[key] = domain_fixup(obj[key])

    return obj


class GroqConversationEntity(ConversationEntity):
    """Represent a conversation entity."""

    _attr_supports_streaming = True

    def __init__(
        self, entry: ConfigEntry[groq.AsyncClient], sub_entry: ConfigSubentry
    ) -> None:
        """Initialize the entity with the API handle."""
        super().__init__()

        self.unique_id = f"{entry.title}_{sub_entry.data.get(CONF_MODEL_IDENTIFIER)}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            entry_type=dr.DeviceEntryType.SERVICE,
            name=f"{entry.title}",
            manufacturer="Groq Cloud Link",
            model=f"{sub_entry.data.get(CONF_MODEL)}",
        )

        self._attr_name = sub_entry.data.get(CONF_MODEL_IDENTIFIER)
        self.chat_history: dict[str, list[Content]] = {}
        self.model = sub_entry.data.get(CONF_MODEL, "llama-3.1-8b-instant")
        self.temperature = sub_entry.data.get(CONF_TEMPERATURE, 1.0)
        self.client: groq.AsyncGroq = entry.runtime_data
        self.language: str | None = None
        self.llm_apis: list[str] = sub_entry.data.get(CONF_LLM_HASS_API, [])
        self.prompt: str = sub_entry.data.get(
            CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
        )
        if sub_entry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Indicate that we support all languages."""
        return MATCH_ALL

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Returns the chat history for use in the UI."""
        return {"chat_history": self.chat_history}

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            intent_response = IntentResponse(
                language=self.language or "English", intent=None
            )

            for _ in range(MAX_TOOL_ITERATIONS):
                async for content in chat_log.async_add_delta_content_stream(
                    agent_id=self.entity_id,
                    stream=self._fullfill_request(
                        chat_log=chat_log, user_input=user_input
                    ),
                ):
                    if isinstance(content, ToolResultContent):
                        pass

                if not chat_log.unresponded_tool_results:
                    break

            # We find the last assistant response for set_speech
            last_message = next(
                (
                    item
                    for item in reversed(chat_log.content)
                    if isinstance(item, AssistantContent)
                ),
                None,
            )
            intent_response.async_set_speech(
                (last_message is not None and (last_message.content or "")) or ""
            )
            self.chat_history[chat_log.conversation_id] = chat_log.content
            return ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=chat_log.continue_conversation,
            )

    async def _fullfill_request(  # noqa: PLR0912
        self, chat_log: ChatLog, user_input: ConversationInput
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """
        Fulfills the request by calling the Groq API.

        Converts the chat log to the format expected by the API.
        """
        await chat_log.async_provide_llm_data(
            user_input.as_llm_context(DOMAIN),
            self.llm_apis,
            self.prompt,
            user_input.extra_system_prompt,
        )

        chat_history: list[ChatCompletionMessageParam] = []

        for message in chat_log.content:
            if isinstance(message, SystemContent):
                chat_history.append(
                    ChatCompletionSystemMessageParam(
                        content=message.content, role="system"
                    )
                )
                continue
            if isinstance(message, UserContent):
                chat_history.append(
                    ChatCompletionUserMessageParam(content=message.content, role="user")
                )
                continue
            if isinstance(message, AssistantContent):
                calls = [
                    ChatCompletionMessageToolCallParam(
                        id=call.id,
                        function=Function(
                            name=call.tool_name, arguments=json.dumps(call.tool_args)
                        ),
                        type="function",
                    )
                    for call in (message.tool_calls or [])
                ]

                chat_history.append(
                    ChatCompletionAssistantMessageParam(
                        content=message.content,
                        role="assistant",
                        tool_calls=calls,
                    )
                )
                continue

            if isinstance(message, ToolResultContent):
                chat_history.append(
                    ChatCompletionToolMessageParam(
                        content=json.dumps(message.tool_result),
                        role="tool",
                        tool_call_id=message.tool_call_id,
                    )
                )
                continue

        tool_definitions: list[FunctionDefinition] = []
        if chat_log.llm_api is not None:
            for tool in chat_log.llm_api.tools:
                parameters: FunctionParameters = convert(
                    tool.parameters,
                    custom_serializer=chat_log.llm_api.custom_serializer,
                )
                tool_definitions.append(
                    FunctionDefinition(
                        name=tool.name,
                        description=tool.description or "Description not available",
                        parameters=parameters,
                    )
                )

        tool_definitions = domain_fixup(tool_definitions)

        tools = [
            ChatCompletionToolParam(function=t, type="function")
            for t in tool_definitions
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_history,
            tools=tools,
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
        )

        chunk: AssistantContentDeltaDict = {"role": "assistant"}
        try:
            async for part in stream:
                for choice in part.choices:
                    if "role" in chunk and choice.delta.content is not None:
                        yield chunk  # Indicate that we're starting a new message.
                        del chunk["role"]
                    chunk["content"] = choice.delta.content
                    chunk["tool_calls"] = []
                    for call in choice.delta.tool_calls or []:
                        if call.function is None or call.function.name is None:
                            continue

                        chunk["tool_calls"].append(
                            llm.ToolInput(
                                id=call.id or ulid_now(),
                                tool_name=call.function.name,
                                tool_args=_fix_invalid_arguments(
                                    call.function.arguments or "{}"
                                ),
                            )
                        )

                    yield chunk
        except groq.APIError as e:
            chunk = {"role": "assistant", "content": f"Error: {e.message}\n\n{e.body}"}
            yield chunk
