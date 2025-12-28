"""Platform for sensor integration."""

from __future__ import annotations

import asyncio
import datetime
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING

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
        hass.data[DOMAIN][entry.entry_id].entities.sensor_entities(),
        update_before_add=True,
    )


class GroqTimeTrackedEntity[T: (int, float)](SensorEntity):
    """Representation of a sensor."""

    def __init__(
        self,
        device: GroqDevice,
        expiry_seconds: float,
        default: T,
        title: str,
        unit: str,
    ) -> None:
        """Initialize the sensor."""
        self.device = device
        self.title = title
        self.unit = unit
        self.default = default
        self._state: T = default
        self.last_update = time.time()
        self.expiry_seconds = expiry_seconds
        self.history: list[tuple[float, T]] = []
        self.listeners: list[Callable[[T], Awaitable[bool]]] = []

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self.title

    @property
    def suggested_object_id(self) -> str | None:
        """Return the unique id of the sensor."""
        return f"{self.device.settings.model}_{self.title}"

    @property
    def unique_id(self) -> str:
        """Return the unique id of the sensor."""
        return (
            f"{self.device.settings.entry_id}_{self.device.settings.model}_{self.title}"
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info from our GroqDevice."""
        return self.device.device_info

    async def track(self, value: T) -> None:
        """Add usage data and queue an update for the expiry of the data."""
        self.history.append((time.time(), value))
        async_call_later(
            hass=self.hass,
            delay=timedelta(seconds=self.expiry_seconds),
            action=self.remove_last_and_update,
        )
        await self.update_value()

    async def remove_last_and_update(self, _: datetime.datetime) -> None:
        """Remove the last item and trigger an update."""
        self.history.pop(0)
        await self.update_value()

    async def update_value(self) -> None:
        """Calculate and publish the current value based on the history."""
        threshold = time.time() - 60
        self._state = reduce(
            lambda prev, tup: ((tup[0] > threshold) and (tup[1] + prev)) or prev,
            self.history,
            self.default,
        )
        self.async_write_ha_state()
        self.listeners = [x for x in self.listeners if not await x(self._state)]

    @property
    def state(self) -> T:
        """Return the state of the sensor."""
        return self._state

    @property
    def unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self.unit

    async def wait_for(
        self, predicate: Callable[[T], Awaitable[bool]] | Callable[[T], bool]
    ) -> asyncio.Event:
        """
        Return an event object that will trigger when the predicate is fulfilled.

        Event flag is set immediately when predicate evaluates to true.
        """
        event = asyncio.Event()

        async def waiter(value: T) -> bool:
            result = predicate(value)
            if isinstance(result, Awaitable):
                result = await result

            if result:
                event.set()
                return True
            return False

        self.listeners.append(waiter)
        await self.update_value()
        return event


class GroqEnumSensor[T: Enum](SensorEntity):
    """Enum text sensor."""

    def __init__(
        self,
        device: GroqDevice,
        title: str,
        initial: T,
    ) -> None:
        """Initialize the sensor."""
        self.device = device
        self.title = title
        self.chosen = initial
        self.options: list[T] = list(initial.__class__.__members__.values())

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self.title

    @property
    def suggested_object_id(self) -> str | None:
        """Return the suggested object id."""
        return f"{self.device.settings.model}_{self.title}"

    @property
    def unique_id(self) -> str:
        """Return a unique ID for the sensor."""
        return f"{self.device.settings.model}_{self.title}"

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info from our GroqDevice."""
        return self.device.device_info

    @property
    def state(self) -> str:
        """Return the current state."""
        return self.chosen.value

    @property
    def extra_state_attributes(self) -> dict[str, list[str]]:
        """Expose the allowed enum options as an attribute."""
        return {"options": [x.value for x in self.options]}

    def set(self, option: T) -> None:
        """Update the state."""
        self.chosen = option
        self.async_write_ha_state()
