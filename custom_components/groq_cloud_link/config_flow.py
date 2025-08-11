"""Config flow for the Groq Cloud Link integration."""

from __future__ import annotations

from typing import Any, TypedDict

import groq
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlowResult,
    ConfigSubentryData,
    ConfigSubentryFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    CONF_AUTH_IDENTIFIER,
    CONF_MODEL_IDENTIFIER,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DOMAIN,
    SUBENTRY_MODEL_PARAMS,
)

API_AUTH_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_AUTH_IDENTIFIER): str,
        vol.Required(CONF_API_KEY): str,
    }
)

MODEL_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MODEL_IDENTIFIER): str,
        vol.Required(CONF_TEMPERATURE, default=1.0): vol.All(
            vol.Coerce(float),
            vol.Range(min=0, max=2, min_included=True, max_included=True),
        ),
        vol.Required(
            CONF_PROMPT, default=llm.DEFAULT_INSTRUCTIONS_PROMPT
        ): TextSelector(TextSelectorConfig(multiline=True)),
    }
)


class ModelParameters(TypedDict):
    """Parameters for AI models."""

    friendly_name: str
    model: str
    temperature: float


class AuthenticationFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for Example integration."""

    VERSION = 1

    def __init__(self) -> None:
        """Create the flow object."""
        self.auth_params: dict[str, Any]

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
        errors: dict[str, str] = {}

        if user_input is None:
            # Ask for authentication (API key)
            return self.async_show_form(data_schema=API_AUTH_SCHEMA, errors=errors)

        if CONF_API_KEY in user_input:
            self.auth_params = user_input

        groq_client: groq.AsyncClient | None = None
        if CONF_MODEL_IDENTIFIER not in user_input:
            # Verify api key and let user select model

            def action() -> groq.AsyncClient | BaseException:
                """Create the groq Client in the executor to prevent blocking."""
                try:
                    return groq.AsyncClient(api_key=user_input[CONF_API_KEY])
                except BaseException as e:  # noqa: BLE001
                    return e

            try:
                groq_client_or_err = await self.hass.async_add_executor_job(action)
                if isinstance(groq_client_or_err, BaseException):
                    raise groq_client_or_err
                groq_client = groq_client_or_err

                model_list = await groq_client.models.list()
                allowed_models = [model.id for model in model_list.data]
            except groq.AuthenticationError:
                errors["base"] = "invalid_auth"
                return self.async_show_form(data_schema=API_AUTH_SCHEMA, errors=errors)

            default_model = (
                "llama-3.1-8b-instant"
                if "llama-3.1-8b-instant" in allowed_models
                else None
            )

            added_models = MODEL_SCHEMA.extend(
                {
                    vol.Required(CONF_MODEL, default=default_model): vol.In(
                        allowed_models
                    )
                }
            )

            added_apis = added_models.extend(
                {
                    vol.Optional(
                        CONF_LLM_HASS_API,
                        description={"suggested_value": []},
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    label=api.name,
                                    value=api.id,
                                )
                                for api in llm.async_get_apis(self.hass)
                            ],
                            multiple=True,
                        )
                    ),
                }
            )
            return self.async_show_form(data_schema=added_apis, errors=errors)
        self._async_abort_entries_match(user_input)

        main_config_data = {
            CONF_AUTH_IDENTIFIER: self.auth_params[CONF_AUTH_IDENTIFIER],
            CONF_API_KEY: self.auth_params[CONF_API_KEY],
        }

        subconfig_data = ConfigSubentryData(
            data={
                CONF_MODEL_IDENTIFIER: user_input[CONF_MODEL_IDENTIFIER],
                CONF_MODEL: user_input[CONF_MODEL],
                CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            },
            subentry_type=SUBENTRY_MODEL_PARAMS,
            title=user_input[CONF_MODEL_IDENTIFIER],
            unique_id=None,
        )

        return self.async_create_entry(
            title=self.auth_params[CONF_AUTH_IDENTIFIER],
            data=main_config_data,
            subentries=[subconfig_data],
        )
