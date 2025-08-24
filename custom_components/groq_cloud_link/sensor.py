"""Platform for sensor integration."""

from __future__ import annotations

import datetime
import time
from datetime import timedelta
from functools import reduce
from typing import TYPE_CHECKING

from groq.types import CompletionUsage
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later

from .const import DOMAIN

if TYPE_CHECKING:
    from . import GroqDevice


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry[GroqDevice],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud Link from a config entry."""
    async_add_entities(
        [hass.data[DOMAIN][entry.entry_id].entities.tokens],
        update_before_add=True,
    )


class UsedTokensEntity(SensorEntity):
    """Representation of a sensor."""

    TOKEN_EXPIRY_TIME_SECONDS = 60

    def __init__(self, device: GroqDevice) -> None:
        """Initialize the sensor."""
        self.device = device
        self._state: int = 0
        self.last_update = time.time()
        self.history: list[tuple[float, int]] = []

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Total tokens"

    @property
    def unique_id(self) -> str:
        """Return the name of the sensor."""
        return self.device.model_parameters.model_identifier

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info from our GroqDevice."""
        return self.device.device_info

    def track(self, usage: CompletionUsage) -> None:
        """Add usage data and queue an update for the expiry of the data."""
        self.history.append((time.time(), usage.total_tokens))
        async_call_later(
            hass=self.hass,
            delay=timedelta(seconds=self.TOKEN_EXPIRY_TIME_SECONDS),
            action=self.remove_last_and_update,
        )
        self.update_value()

    async def remove_last_and_update(self, _: datetime.datetime) -> None:
        """Remove the last item and trigger an update."""
        self.history.pop(0)
        self.update_value()

    def update_value(self) -> None:
        """Calculate and publish the current value based on the history."""
        threshold = time.time() - 60
        self._state = reduce(
            lambda prev, tup: ((tup[0] > threshold) and (tup[1] + prev)) or prev,
            self.history,
            0,
        )
        self.async_write_ha_state()

    @property
    def state(self) -> int:
        """Return the state of the sensor."""
        return self._state

    @property
    def unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return "Tokens"
