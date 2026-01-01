"""Config flow for the Groq Cloud Link integration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import groq
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.conversation.const import ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry, ConfigEntryBaseFlow, ConfigFlowResult, ConfigSubentryFlow
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.llm import API, AssistAPI, async_get_apis
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    BROWSER_SEARCH_MODELS,
    CODE_INTERPRETER_MODELS,
    CONF_BASE_URL,
    CONF_FEATURES,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TEXT_TO_SPEECH_MODEL,
    CONF_TEXT_TO_SPEECH_VOICE,
    DOMAIN,
    TOOL_USE_MODELS,
)
from .features import LLMFeatures

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from types import MappingProxyType


@dataclass
class GroqDeviceSettings:
    """Holds all parameters to create a GroqDevice."""

    api_key: str = ""
    base_url: str | None = None
    model: str = "openai/gpt-oss-20b"
    prompt: str = llm.DEFAULT_INSTRUCTIONS_PROMPT
    temperature: float = 1.0
    voice: str = "troy"
    tts_model: str = "canopylabs/orpheus-v1-english"
    stt_model: str = "whisper-large-v3"
    stt_temperature: float = 0.2
    _llm_apis: list[str] = field(default_factory=list)
    conversation_features: int = 0
    features: list[LLMFeatures] = field(
        default_factory=lambda: [LLMFeatures.ALLOW_BROWSER_SEARCH, LLMFeatures.ALLOW_CODE_EXECUTION],
        metadata={"serialize": False},
    )
    entry_id: str = ""

    async def create(self, hass: HomeAssistant) -> groq.AsyncGroq:
        """Create the client asynchronously through the hass executor."""
        return await hass.async_add_executor_job(lambda: groq.AsyncClient(api_key=self.api_key, base_url=self.base_url))

    def serialize(self) -> dict[str, str]:
        """Prepare settings for disk storage."""
        cloned = {}
        for f in fields(self):
            if f.metadata.get("serialize", True):
                cloned[f.name] = getattr(self, f.name)
        cloned["features"] = [x.name for x in self.features]
        return cloned

    def get_apis(self, hass: HomeAssistant) -> list[API]:
        """Get the APIs selected."""
        return [api for api in async_get_apis(hass) if api.id in self._llm_apis]

    def set_apis(self, apis: list[API]) -> None:
        """Set the APIs selected."""
        self._llm_apis = [api.id for api in apis]
        for api in apis:
            if isinstance(api, AssistAPI):
                self.conversation_features = ConversationEntityFeature.CONTROL
                break

    @staticmethod
    def unserialize(obj: MappingProxyType[str, Any], entry_id: str) -> GroqDeviceSettings:
        """Read settings from disk storage."""
        obj2 = dict(obj)
        obj2["features"] = [LLMFeatures[name] for name in obj["features"] if name in LLMFeatures.__members__]
        zelf = GroqDeviceSettings(**obj2)
        zelf.entry_id = entry_id
        return zelf

    @staticmethod
    def default() -> GroqDeviceSettings:
        """Create sane default settings."""
        return GroqDeviceSettings()

    async def generate_ui(
        self, form: ConfigEntryBaseFlow, input_getter: Callable[[Any], Any | None]
    ) -> AsyncGenerator[ConfigFlowResult | bool, Any]:
        """Generate the user interface used in setup as generator."""
        yield form.async_show_form(
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY, default=self.api_key): str,
                    vol.Optional(CONF_BASE_URL, self.base_url): str,
                }
            )
        )

        api_key: str | None = input_getter(CONF_API_KEY)
        if api_key is None:
            yield form.async_abort(reason="Missing API key")
            return

        self.api_key = api_key
        self.base_url = input_getter(CONF_BASE_URL)

        try:
            client: groq.AsyncClient = await self.create(form.hass)

            model_list = await client.models.list()
            allowed_models = [model.id for model in model_list.data]

            default_model = "openai/gpt-oss-20b" if "openai/gpt-oss-20b" in allowed_models else None

            model_schema = vol.Schema(
                {
                    vol.Required(CONF_TEMPERATURE, default=self.temperature): vol.All(
                        vol.Coerce(float),
                        vol.Range(min=0, max=2, min_included=True, max_included=True),
                    ),
                    vol.Required(CONF_PROMPT, default=self.prompt): TextSelector(TextSelectorConfig(multiline=True)),
                }
            )

            model_schema = model_schema.extend(
                {vol.Required(CONF_MODEL, default=self.model or default_model): vol.In(allowed_models)}
            )

            allowed_apis = llm.async_get_apis(form.hass)
            model_schema = model_schema.extend(
                {
                    vol.Required(
                        CONF_LLM_HASS_API,
                        default=[x.id for x in self.get_apis(form.hass)],
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    label=api.name,
                                    value=api.id,
                                )
                                for api in allowed_apis
                            ],
                            multiple=True,
                        )
                    ),
                }
            ).extend(
                {
                    vol.Required(
                        CONF_FEATURES,
                        default=[x.name for x in self.features],
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    label=x.value,
                                    value=x.name,
                                )
                                for x in LLMFeatures.__members__.values()
                            ],
                            multiple=True,
                        )
                    ),
                }
            )
            yield form.async_show_form(data_schema=model_schema)

            self.temperature = input_getter(CONF_TEMPERATURE) or self.temperature
            self.prompt = input_getter(CONF_PROMPT) or self.prompt
            self.model = input_getter(CONF_MODEL) or self.model

            self.set_apis([api for api in allowed_apis if api.id in (input_getter(CONF_LLM_HASS_API) or [])])

            if self.model not in TOOL_USE_MODELS and len(self._llm_apis) > 0:
                yield form.async_abort(reason="Model does not support tool use.")
                return

            self.features = [
                LLMFeatures[name] for name in (input_getter(CONF_FEATURES) or []) if name in LLMFeatures.__members__
            ]

            if self.model not in BROWSER_SEARCH_MODELS and LLMFeatures.ALLOW_BROWSER_SEARCH in self.features:
                yield form.async_abort(reason="Model does not support browser search.")
                return

            if self.model not in CODE_INTERPRETER_MODELS and LLMFeatures.ALLOW_CODE_EXECUTION in self.features:
                yield form.async_abort(reason="Model does not support code execution.")
                return

            voice_schema = vol.Schema(
                {
                    vol.Required(CONF_TEXT_TO_SPEECH_MODEL, default=self.tts_model): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    label=model,
                                    value=model,
                                )
                                for model in [
                                    "canopylabs/orpheus-v1-english",
                                    "canopylabs/orpheus-arabic-saudi",
                                ]
                            ],
                            multiple=False,
                        )
                    ),
                    vol.Required(CONF_TEXT_TO_SPEECH_VOICE, default=self.voice): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    label=voice,
                                    value=voice,
                                )
                                for voice in [
                                    "autumn",
                                    "diana",
                                    "hannah",
                                    "austin",
                                    "daniel",
                                    "troy",
                                ]
                            ],
                            multiple=False,
                        )
                    ),
                }
            )
            yield form.async_show_form(data_schema=voice_schema)
            self.voice = input_getter(CONF_TEXT_TO_SPEECH_VOICE) or self.voice
            yield True
        except groq.APIError as e:
            yield form.async_abort(reason=e.message)
            yield False
        except groq.GroqError as e:
            yield form.async_abort(reason=f"An unknown error has occurred: {e}")
            yield False


class AuthenticationFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handles the user input for setting up a Groq device."""

    VERSION = 2

    def __init__(self) -> None:
        """Create the flow object."""
        self.settings = GroqDeviceSettings.default()
        self.state = self.settings.generate_ui(self, lambda x: self.user_input.get(x))
        self.user_input: dict[str, Any] = {}

    @classmethod
    @callback
    def async_get_supported_subentry_types(cls, _config_entry: ConfigEntry) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry):
        return GroqOptionsFlow(config_entry)

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step."""
        self.user_input = user_input or {}
        result = await anext(self.state)
        if result is True:
            return self.async_create_entry(
                title="Groq Cloud",
                data={},
                options=self.settings.serialize(),
                subentries=[],
            )
        if result is False:
            return self.async_abort(reason=f"The data you entered yielded no usable configuration: {self.user_input}")
        return result


class GroqOptionsFlow(config_entries.OptionsFlowWithReload):
    def __init__(self, config_entry: ConfigEntry) -> None:
        self.user_input: dict[str, Any] = {}
        self.settings = GroqDeviceSettings.unserialize(config_entry.options, config_entry.entry_id)
        self.state = self.settings.generate_ui(self, lambda x: self.user_input.get(x))

    async def async_step_init(self, user_input: dict[str, Any] | None) -> ConfigFlowResult:
        self.user_input = user_input or {}
        result = await anext(self.state)
        if result is True:
            return self.async_create_entry(title=self.config_entry.title, data=self.settings.serialize())
        if result is False:
            return self.async_abort(reason=f"The data you entered yielded no usable configuration: {self.user_input}")
        return result
