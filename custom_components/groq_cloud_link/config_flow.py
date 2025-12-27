"""Config flow for the Groq Cloud Link integration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import groq
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlowResult,
    ConfigSubentryFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.llm import API, async_get_apis
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    CONF_BASE_URL,
    CONF_FEATURES,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_VOICE,
    DOMAIN,
)
from .features import LLMFeatures

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass
class GroqDeviceSettings:
    """Holds all parameters to create a GroqDevice."""

    api_key: str = ""
    base_url: str | None = None
    model: str = "llama-3.1-8b-instant"
    prompt: str = ""
    temperature: float = 1.0
    voice: str = "troy"
    llm_apis: list[API] = field(default_factory=list)
    features: list[LLMFeatures] = field(default_factory=list)
    entry_id: str = ""

    async def create(self, hass: HomeAssistant) -> groq.AsyncGroq:
        """Create the client asynchronously through the hass executor."""
        return await hass.async_add_executor_job(
            lambda: groq.AsyncClient(api_key=self.api_key, base_url=self.base_url)
        )

    def serialize(self) -> dict[str, str]:
        """Prepare settings for disk storage."""
        cloned = dataclasses.asdict(self)
        cloned["features"] = [x.name for x in cloned["features"]]
        cloned["llm_apis"] = [x.id for x in cloned["llm_apis"]]
        return cloned

    @staticmethod
    def unserialize(
        hass: HomeAssistant, obj: MappingProxyType[str, Any]
    ) -> GroqDeviceSettings:
        """Read settings from disk storage."""
        obj2 = dict(obj)
        obj2["features"] = [
            LLMFeatures[name]
            for name in obj["features"]
            if name in LLMFeatures.__members__
        ]
        obj2["llm_apis"] = [
            api for api in async_get_apis(hass) if api.id in obj2["llm_apis"]
        ]
        return GroqDeviceSettings(**obj2)


class AuthenticationFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handles the user input for setting up a Groq device."""

    VERSION = 1

    def __init__(self) -> None:
        """Create the flow object."""
        self.state = self.__gen()
        self.user_input: dict[str, Any] = {}

    async def __gen(self) -> AsyncGenerator[ConfigFlowResult, Any]:
        yield self.async_show_form(
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): str,
                    vol.Optional(CONF_BASE_URL): str,
                }
            )
        )
        settings = GroqDeviceSettings()

        api_key: str | None = self.user_input.get(CONF_API_KEY)
        if api_key is None:
            yield self.async_abort(reason="Missing API key")
            return

        settings.api_key = api_key
        settings.base_url = self.user_input.get(CONF_BASE_URL)

        try:
            client: groq.AsyncClient = await settings.create(self.hass)

            model_list = await client.models.list()
            allowed_models = [model.id for model in model_list.data]

            default_model = (
                "llama-3.1-8b-instant"
                if "llama-3.1-8b-instant" in allowed_models
                else None
            )

            model_schema = vol.Schema(
                {
                    vol.Required(CONF_TEMPERATURE, default=1.0): vol.All(
                        vol.Coerce(float),
                        vol.Range(min=0, max=2, min_included=True, max_included=True),
                    ),
                    vol.Required(
                        CONF_PROMPT, default=llm.DEFAULT_INSTRUCTIONS_PROMPT
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                }
            )

            model_schema = model_schema.extend(
                {
                    vol.Required(CONF_MODEL, default=default_model): vol.In(
                        allowed_models
                    )
                }
            )

            allowed_apis = llm.async_get_apis(self.hass)
            model_schema = model_schema.extend(
                {
                    vol.Required(
                        CONF_LLM_HASS_API,
                        description={"suggested_value": []},
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
                        description={
                            "suggested_value": [
                                LLMFeatures.ALLOW_BROWSER_SEARCH.name,
                                LLMFeatures.ALLOW_CODE_EXECUTION.name,
                            ]
                        },
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
            yield self.async_show_form(data_schema=model_schema)

            settings.temperature = self.user_input[CONF_TEMPERATURE]
            settings.prompt = self.user_input[CONF_PROMPT]
            settings.model = self.user_input[CONF_MODEL]
            settings.llm_apis = [
                x
                for x in self.user_input.get(CONF_LLM_HASS_API, [])
                if x in allowed_apis
            ]

            settings.features = [
                LLMFeatures[name]
                for name in self.user_input.get(CONF_FEATURES, [])
                if name in LLMFeatures.__members__
            ]

            voice_schema = vol.Schema(
                {
                    vol.Required(CONF_VOICE, default="troy"): SelectSelector(
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
                    )
                }
            )
            yield self.async_show_form(data_schema=voice_schema)
            settings.voice = self.user_input[CONF_VOICE]
            yield self.async_create_entry(
                title="Groq Cloud",
                data=settings.serialize(),
                subentries=[],
            )
        except groq.APIError as e:
            yield self.async_abort(reason=e.message)
        except groq.GroqError as e:
            yield self.async_abort(reason=f"An unknown error has occurred: {e}")

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, _config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        self.user_input = user_input or {}
        try:
            return await anext(self.state)
        except StopAsyncIteration:
            return self.async_abort(
                reason=f"The data you entered yielded no usable configuration: {self.user_input}"
            )


#
# class __AuthenticationFlowOLD(config_entries.ConfigFlow, domain=DOMAIN):
#     """Config flow for Example integration."""
#
#     VERSION = 1
#
#     def __init__(self) -> None:
#         """Create the flow object."""
#         self.auth_params: dict[str, Any]
#
#     @classmethod
#     @callback
#     def async_get_supported_subentry_types(
#         cls, _config_entry: ConfigEntry
#     ) -> dict[str, type[ConfigSubentryFlow]]:
#         """Return subentries supported by this integration."""
#         return {}
#
#     async def async_step_user(
#         self, user_input: dict[str, Any] | None = None
#     ) -> ConfigFlowResult:
#         """Handle the initial step."""
#         errors: dict[str, str] = {}
#
#         if user_input is None:
#             # Ask for authentication (API key)
#             return self.async_show_form(data_schema=API_AUTH_SCHEMA, errors=errors)
#
#         if CONF_API_KEY in user_input:
#             self.auth_params = user_input
#
#         groq_client: groq.AsyncClient | None = None
#         if CONF_API_KEY not in user_input:
#             # Verify api key and let user select model
#
#             def action() -> groq.AsyncClient | BaseException:
#                 """Create the groq Client in the executor to prevent blocking."""
#                 try:
#                     return groq.AsyncClient(api_key=user_input[CONF_API_KEY])
#                 except BaseException as e:
#                     return e
#
#             try:
#                 groq_client_or_err = await self.hass.async_add_executor_job(action)
#                 if isinstance(groq_client_or_err, BaseException):
#                     raise groq_client_or_err
#                 groq_client = groq_client_or_err
#
#                 model_list = await groq_client.models.list()
#                 allowed_models = [model.id for model in model_list.data]
#             except groq.AuthenticationError:
#                 errors["base"] = "invalid_auth"
#                 return self.async_show_form(data_schema=API_AUTH_SCHEMA, errors=errors)
#
#             default_model = (
#                 "llama-3.1-8b-instant"
#                 if "llama-3.1-8b-instant" in allowed_models
#                 else None
#             )
#
#             added_models = MODEL_SCHEMA.extend(
#                 {
#                     vol.Required(CONF_MODEL, default=default_model): vol.In(
#                         allowed_models
#                     )
#                 }
#             )
#
#             model_params = added_models.extend(
#                 {
#                     vol.Optional(
#                         CONF_LLM_HASS_API,
#                         description={"suggested_value": []},
#                     ): SelectSelector(
#                         SelectSelectorConfig(
#                             options=[
#                                 SelectOptionDict(
#                                     label=api.name,
#                                     value=api.id,
#                                 )
#                                 for api in llm.async_get_apis(self.hass)
#                             ],
#                             multiple=True,
#                         )
#                     ),
#                 }
#             ).extend(
#                 {
#                     vol.Optional(
#                         CONF_FEATURES,
#                         description={
#                             "suggested_value": [
#                                 LLMFeatures.ALLOW_BROWSER_SEARCH.name,
#                                 LLMFeatures.ALLOW_CODE_EXECUTION.name,
#                             ]
#                         },
#                     ): SelectSelector(
#                         SelectSelectorConfig(
#                             options=[
#                                 SelectOptionDict(
#                                     label=x.value,
#                                     value=x.name,
#                                 )
#                                 for x in LLMFeatures.__members__.values()
#                             ],
#                             multiple=True,
#                         )
#                     ),
#                 }
#             )
#             return self.async_show_form(data_schema=model_params, errors=errors)
#         self._async_abort_entries_match(user_input)
#
#         main_config_data = {
#             CONF_API_KEY: self.auth_params[CONF_API_KEY],
#             SUBENTRY_MODEL_PARAMS: {
#                 CONF_MODEL: user_input[CONF_MODEL],
#                 CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
#                 CONF_PROMPT: user_input[CONF_PROMPT],
#                 CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
#                 CONF_FEATURES: user_input[CONF_FEATURES],
#             },
#         }
#         return self.async_create_entry(
#             title=self.auth_params[CONF_AUTH_IDENTIFIER],
#             data=main_config_data,
#             subentries=[],
#         )
#
