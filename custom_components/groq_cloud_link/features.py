"""Feature flags for the LLMs."""

from enum import Enum


class LLMFeatures(Enum):
    """The reason for being rate limited."""

    ALLOW_BROWSER_SEARCH = "Browser Search"
    ALLOW_SEARCH_WITH_LIVE_DATA = "Alow browser search with live data."
    ALLOW_CODE_EXECUTION = "Allow code execution"
