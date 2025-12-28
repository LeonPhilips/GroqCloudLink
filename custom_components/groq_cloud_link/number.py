"""Platform for number integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from homeassistant.components.number import NumberEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from . import GroqDevice


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry[GroqDevice],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud Link number entities from a config entry."""
    async_add_entities(
        hass.data[DOMAIN][entry.entry_id].entities.number_entities(),
        update_before_add=True,
    )


class GroqNumberEntity[T: (int, float)](NumberEntity, RestoreEntity):
    """Representation of an input number entity."""

    def __init__(  # noqa: PLR0913
        self,
        device: GroqDevice,
        title: str,
        unit: str,
        min_value: T,
        max_value: T,
        step: T,
        initial: float,
        on_change: Callable[[T], Awaitable[None]] | None,
    ) -> None:
        """Initialize the number entity."""
        self.device = device
        self.title = title
        self.initial = initial
        self.on_change: Callable[[T], Awaitable[None]] = on_change or self.__noop
        self._attr_native_min_value = min_value
        self._attr_native_max_value = max_value
        self._attr_native_step = step
        self._attr_native_unit_of_measurement = unit
        self._attr_native_value: float | None = None

    async def async_added_to_hass(self) -> None:
        """Restore the previous setting or set to initial value."""
        await super().async_added_to_hass()
        if (last_state := await self.async_get_last_state()) is not None:
            try:
                self._attr_native_value = float(last_state.state)
            except ValueError:
                self._attr_native_value = self.initial
        else:
            self._attr_native_value = self.initial

    async def __noop(self, _: T) -> None:
        """Do nothing."""

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self.title

    @property
    def suggested_object_id(self) -> str | None:
        """Return the suggested object id."""
        return f"{self.device.settings.model}_{self.title}"

    @property
    def unique_id(self) -> str:
        """Return a unique ID for the entity."""
        return (
            f"{self.device.settings.entry_id}_{self.device.settings.model}_{self.title}"
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info from our GroqDevice."""
        return self.device.device_info

    @property
    def native_value(self) -> float | None:
        """Return the current value."""
        return self._attr_native_value

    def get_value(self) -> T:
        """Get the current value as T."""
        return cast("T", self._attr_native_value)

    async def async_set_native_value(self, value: T) -> None:
        """Handle user updating the value."""
        self._attr_native_value = value
        self.async_write_ha_state()
        await self.on_change(value)
