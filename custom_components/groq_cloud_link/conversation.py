"""Conversation support for Groq Cloud."""

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

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
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse
from homeassistant.helpers.llm import AssistAPI
from homeassistant.util.ulid import ulid_now
from voluptuous_openapi import convert

if TYPE_CHECKING:
    from . import GroqDevice

from .const import (
    DOMAIN,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry["GroqDevice"],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud Link from a config entry."""
    async_add_entities(
        hass.data[DOMAIN][entry.entry_id].entities.conversation_entities(),
        update_before_add=True,
    )


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

    def __init__(self, device: "GroqDevice") -> None:
        """Initialize the entity with the API handle."""
        super().__init__()
        self.device = device
        self.language: str | None = None

        for api in device.model_parameters.apis:
            if isinstance(api, AssistAPI):
                self._attr_supported_features = ConversationEntityFeature.CONTROL
                break

    @property
    def name(self) -> str:
        """Return the name of the conversation Entity."""
        return self.device.model_parameters.model_identifier

    @property
    def unique_id(self) -> str:
        """Return the name of the sensor."""
        return self.device.model_parameters.model_identifier

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info from our GroqDevice."""
        return self.device.device_info

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Indicate that we support all languages."""
        return MATCH_ALL

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
            [api.id for api in self.device.model_parameters.apis],
            self.device.model_parameters.prompt,
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

        await self.device.add_request()
        stream = await self.device.get_client().chat.completions.create(
            model=self.device.model_parameters.model,
            messages=chat_history,
            tools=tools,
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
            extra_body={"stream_options": {"include_usage": True}},
        )

        chunk: AssistantContentDeltaDict = {"role": "assistant"}
        try:
            async for part in stream:
                if part.usage is not None:
                    await self.device.track_usage(part.usage)

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
