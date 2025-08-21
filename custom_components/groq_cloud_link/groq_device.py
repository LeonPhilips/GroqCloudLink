"""GroqDevice represents all entities the device has."""

import groq
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .conversation import GroqConversationEntity


class GroqDevice:
    """The device that represents one model that executes."""

    def __init__(
        self, entry: ConfigEntry[groq.AsyncClient], sub_entry: ConfigSubentry
    ) -> None:
        """Initialize all relevant entities for the device."""
        self.entry = entry
        self.sub_entry = sub_entry

    async def async_setup_entry(
        self,
        hass: HomeAssistant,
        async_add_entities: AddConfigEntryEntitiesCallback,
    ) -> None:
        """Set up Groq Cloud Link from a config entry."""

        def action() -> groq.AsyncClient | BaseException:
            """Create the groq Client in the executor to prevent blocking."""
            try:
                return groq.AsyncClient(api_key=self.entry.data[CONF_API_KEY])
            except BaseException as e:  # noqa: BLE001
                return e

        groq_client_or_err = await hass.async_add_executor_job(action)
        if isinstance(groq_client_or_err, BaseException):
            raise groq_client_or_err
        self.entry.runtime_data = groq_client_or_err

        async_add_entities(
            [GroqConversationEntity(self.entry, self.sub_entry)],
            config_subentry_id=self.sub_entry.subentry_id,
        )
