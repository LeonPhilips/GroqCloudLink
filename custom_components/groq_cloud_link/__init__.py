"""The Groq Cloud Link integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import groq
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.llm import API, async_get_apis

from .const import (
    CONF_MODEL_IDENTIFIER,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DOMAIN,
    SUBENTRY_MODEL_PARAMS,
)
from .conversation import GroqConversationEntity
from .sensor import UsedTokensEntity

PLATFORMS = [Platform.CONVERSATION, Platform.SENSOR]


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry[GroqDevice]
) -> bool:
    """Set up Groq Cloud Link from a config entry."""
    device = GroqDevice(entry)
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
    allow_search: bool = False


@dataclass
class Entities:
    """The entities that the GroqDevice has references to."""

    conversation: GroqConversationEntity
    tokens: UsedTokensEntity


class GroqDevice:
    """Holds relevant data for multiple entities for the same model."""

    def __init__(self, entry: ConfigEntry[GroqDevice]) -> None:
        """Create the home assistant device. Holds data stored in entries."""
        self.api_key: str = entry.data[CONF_API_KEY]
        self.entry_id = entry.entry_id
        self.cache = {"apis": entry.data[SUBENTRY_MODEL_PARAMS][CONF_LLM_HASS_API]}
        self.model_parameters = ModelParameters(
            model_identifier=entry.data[SUBENTRY_MODEL_PARAMS][CONF_MODEL_IDENTIFIER],
            model=entry.data[SUBENTRY_MODEL_PARAMS][CONF_MODEL],
            temperature=entry.data[SUBENTRY_MODEL_PARAMS][CONF_TEMPERATURE],
            apis=[],
            prompt=entry.data[SUBENTRY_MODEL_PARAMS][CONF_PROMPT],
            allow_search=False,
        )
        self.client: groq.AsyncClient | None = None
        self.device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{self.entry_id}")},
            entry_type=DeviceEntryType.SERVICE,
            name=f"{self.model_parameters.model_identifier}",
            manufacturer="Groq Cloud Link",
            model=f"{self.model_parameters.model}",
        )
        self.entities: Entities | None = None

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
        api_names: list[str] = self.cache.pop("apis")
        for api in async_get_apis(hass):
            if api.id in api_names:
                self.model_parameters.apis.append(api)

        self.entities = Entities(
            conversation=GroqConversationEntity(self),
            tokens=UsedTokensEntity(self),
        )

    async def track_usage(self, usage: groq.types.CompletionUsage) -> None:
        """Tracks usage. Called from the conversation entity."""
        if self.entities is None:
            msg = "Prepare not called before accessing track_usage()"
            raise AssertionError(msg)
        self.entities.tokens.track(usage)

    def get_entities(self) -> Entities:
        """Add all relevant entities for this device."""
        if self.entities is None:
            msg = "Prepare not called before accessing get_entities()"
            raise AssertionError(msg)

        return self.entities
