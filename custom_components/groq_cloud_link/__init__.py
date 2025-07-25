"""The Groq Cloud Link integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .conversation import GroqConversationEntity

PLATFORMS = [Platform.CONVERSATION]


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry[GroqConversationEntity]
) -> bool:
    """Set up Groq Cloud Link from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True
