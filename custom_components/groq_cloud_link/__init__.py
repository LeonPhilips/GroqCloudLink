"""The Groq Cloud Link integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import groq
from homeassistant.components.conversation.chat_log import AssistantContentDeltaDict
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.llm import API, async_get_apis

from .const import (
    CONF_FEATURES,
    CONF_MODEL_IDENTIFIER,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DOMAIN,
    SUBENTRY_MODEL_PARAMS,
)
from .conversation import GroqConversationEntity
from .features import LLMFeatures
from .number import GroqNumberEntity
from .sensor import GroqEnumSensor, GroqTimeTrackedEntity

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Awaitable, Callable

PLATFORMS = [Platform.CONVERSATION, Platform.SENSOR, Platform.NUMBER]


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry[GroqDevice]
) -> bool:
    """Set up Groq Cloud Link from a config entry."""
    device = GroqDevice(hass, entry)
    await device.prepare(hass)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = device
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok


@dataclass
class ModelParameters:
    """Describes model settings."""

    model_identifier: str
    model: str
    prompt: str
    temperature: float
    apis: list[API] = field(default_factory=list)
    features: list[LLMFeatures] = field(default_factory=list)


@dataclass
class Entities:
    """The entities that the GroqDevice has references to."""

    conversation: GroqConversationEntity
    tokens: GroqTimeTrackedEntity[int]
    requests: GroqTimeTrackedEntity[int]

    device_status: GroqEnumSensor[GroqDeviceState]
    rate_limit_reason: GroqEnumSensor[GroqDeviceRateLimitReason]

    max_tokens_per_min: GroqNumberEntity[int]
    max_requests_per_min: GroqNumberEntity[int]

    def sensor_entities(self) -> list[Entity]:
        """Return all entities of the Sensor type."""
        return [self.tokens, self.requests, self.device_status, self.rate_limit_reason]

    def conversation_entities(self) -> list[Entity]:
        """Return all entities of the Conversation type."""
        return [self.conversation]

    def number_entities(self) -> list[Entity]:
        """Return all entities of the Conversation type."""
        return [self.max_tokens_per_min, self.max_requests_per_min]


class GroqDeviceState(Enum):
    """The device state of the Groq device."""

    READY = "READY"
    PROCESSING = "PROCESSING"
    RATE_LIMITED = "RATE_LIMITED"


class GroqDeviceRateLimitReason(Enum):
    """The reason for being rate limited."""

    REASON_NONE = "None"
    REASON_TOO_MANY_REQUESTS = "Request count"
    REASON_TOO_MANY_TOKENS = "Token usage"


class GroqDevice:
    """Holds relevant data for multiple entities for the same model."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry[GroqDevice]) -> None:
        """Create the home assistant device. Holds data stored in entries."""
        self.api_key: str = entry.data[CONF_API_KEY]
        self.client: groq.AsyncClient | None = None
        self.entry_id = entry.entry_id
        self.model_parameters = ModelParameters(
            model_identifier=entry.data[SUBENTRY_MODEL_PARAMS][CONF_MODEL_IDENTIFIER],
            model=entry.data[SUBENTRY_MODEL_PARAMS][CONF_MODEL],
            temperature=entry.data[SUBENTRY_MODEL_PARAMS][CONF_TEMPERATURE],
            apis=[
                api
                for api in async_get_apis(hass)
                if api.id in entry.data[SUBENTRY_MODEL_PARAMS][CONF_LLM_HASS_API]
            ],
            prompt=entry.data[SUBENTRY_MODEL_PARAMS][CONF_PROMPT],
            features=[
                feature
                for feature in LLMFeatures.__members__.values()
                if feature.name in entry.data[SUBENTRY_MODEL_PARAMS][CONF_FEATURES]
            ],
        )
        self.device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{self.entry_id}")},
            entry_type=DeviceEntryType.SERVICE,
            name=f"{self.model_parameters.model_identifier}",
            manufacturer="Groq Cloud Link",
            model=f"{self.model_parameters.model}",
        )
        self.entities: Entities = Entities(
            conversation=GroqConversationEntity(self),
            tokens=GroqTimeTrackedEntity(
                self,
                expiry_seconds=60,
                default=0,
                title="Token usage",
                unit="Tokens/min",
            ),
            requests=GroqTimeTrackedEntity(
                self,
                expiry_seconds=60,
                default=0,
                title="Request usage",
                unit="Requests/min",
            ),
            max_tokens_per_min=GroqNumberEntity(
                self,
                title="Max tokens/min",
                unit="Tokens/min",
                min_value=1,
                max_value=500000,
                step=1,
                initial=5000,
                on_change=None,
            ),
            max_requests_per_min=GroqNumberEntity(
                self,
                title="Max requests/min",
                unit="Requests/min",
                min_value=1,
                max_value=600,
                step=1,
                initial=30,
                on_change=None,
            ),
            rate_limit_reason=GroqEnumSensor(
                self,
                title="Rate limit reason",
                initial=GroqDeviceRateLimitReason.REASON_NONE,
            ),
            device_status=GroqEnumSensor(
                self,
                title="State",
                initial=GroqDeviceState.READY,
            ),
        )

    def __action(self) -> groq.AsyncClient | BaseException:
        """Create the groq Client in the executor to prevent blocking."""
        try:
            return groq.AsyncClient(api_key=self.api_key)
        except BaseException as e:  # noqa: BLE001
            return e

    def get_client(self) -> groq.AsyncClient:
        """Obtain a client handle, throws if `prepare` isn't called."""
        if self.client is None:
            msg = "Prepare not called before accessing get_client()"
            raise AssertionError(msg)
        return self.client

    async def prepare(self, hass: HomeAssistant) -> None:
        """Prepare the groq api client."""
        if self.client is not None:
            return

        groq_client_or_err = await hass.async_add_executor_job(self.__action)
        if isinstance(groq_client_or_err, BaseException):
            raise groq_client_or_err
        self.client = groq_client_or_err

    async def track_usage(self, usage: groq.types.CompletionUsage) -> None:
        """Tracks usage. Called from the conversation entity."""
        await self.entities.tokens.track(usage.total_tokens)

    async def add_request(self) -> None:
        """Tracks the request count. Called from the conversation entity."""
        await self.entities.requests.track(1)

    async def pre_request_wait(
        self, message_callback: Callable[[str], Awaitable[AssistantContentDeltaDict]]
    ) -> AsyncGenerator[AssistantContentDeltaDict | None]:
        """
        Wait for the usage to fall within limits and set the device state to PROCESSING.

        This is a small wrapper for `wait_for_limits` that also sets device status.
        """
        async for msg in self.wait_for_limits(message_callback):
            yield msg
        # All is OK. Let's go.
        self.entities.device_status.set(GroqDeviceState.PROCESSING)
        yield None

    async def post_request_wait(
        self, _message_callback: Callable[[str], Awaitable[AssistantContentDeltaDict]]
    ) -> AsyncGenerator[AssistantContentDeltaDict | None]:
        """Set the device state back to READY."""
        self.entities.device_status.set(GroqDeviceState.READY)
        yield None

    async def wait_for_limits(
        self, message_callback: Callable[[str], Awaitable[AssistantContentDeltaDict]]
    ) -> AsyncGenerator[AssistantContentDeltaDict | None]:
        """Wait for the usage to fall within limits and return."""
        # Wait for request limits
        request_below_rate_limit_event: asyncio.Event = (
            await self.entities.requests.wait_for(
                lambda requests: requests
                < self.entities.max_requests_per_min.get_value()
            )
        )
        if not request_below_rate_limit_event.is_set():
            self.entities.device_status.set(GroqDeviceState.RATE_LIMITED)
            self.entities.rate_limit_reason.set(
                GroqDeviceRateLimitReason.REASON_TOO_MANY_REQUESTS
            )
            yield await message_callback("Waiting for request quota...")
            await request_below_rate_limit_event.wait()

        # Wait for token limits
        tokens_below_rate_limit_event: asyncio.Event = (
            await self.entities.tokens.wait_for(
                lambda tokens: tokens < self.entities.max_tokens_per_min.get_value()
            )
        )
        if not tokens_below_rate_limit_event.is_set():
            self.entities.device_status.set(GroqDeviceState.RATE_LIMITED)
            self.entities.rate_limit_reason.set(
                GroqDeviceRateLimitReason.REASON_TOO_MANY_TOKENS
            )
            yield await message_callback("Waiting for token quota...")
            await tokens_below_rate_limit_event.wait()

        self.entities.rate_limit_reason.set(GroqDeviceRateLimitReason.REASON_NONE)
        yield None

    def get_entities(self) -> Entities:
        """Add all relevant entities for this device."""
        return self.entities
