"""Errors for the Groq Cloud Link integration."""

from homeassistant.exceptions import HomeAssistantError


class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuthError(HomeAssistantError):
    """Error to indicate there is invalid auth."""
