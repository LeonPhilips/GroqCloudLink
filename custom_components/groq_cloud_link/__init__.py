"""The Groq Cloud Link integration."""

from __future__ import annotations

import groq
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.groq_cloud_link.groq_device import GroqDevice

PLATFORMS = [Platform.CONVERSATION]

from const import SUBENTRY_MODEL_PARAMS


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry[groq.AsyncClient]
) -> bool:
    """Set up Groq Cloud Link from a config entry."""

    for sub_entry in entry.subentries.values():
        if sub_entry.subentry_type != SUBENTRY_MODEL_PARAMS:
            continue

        device = GroqDevice(entry, sub_entry)
        hass.config_entries.async_forward_entry_setups()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True
