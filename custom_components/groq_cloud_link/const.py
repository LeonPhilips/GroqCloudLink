"""Constants for the Groq Cloud Link integration."""

DOMAIN = "groq_cloud_link"


CONF_TEMPERATURE = "temperature"
CONF_AUTH_IDENTIFIER = "authentication_identifier"
CONF_MODEL_IDENTIFIER = "model_identifier"
CONF_PROMPT = "prompt"
CONF_FEATURES = "features"
CONF_TEXT_TO_SPEECH_VOICE = "voice"
CONF_TEXT_TO_SPEECH_MODEL = "voice_model"
CONF_SPEECH_TO_TEXT_MODEL = "speech_to_text_model"
CONF_BASE_URL = "base_url"

SUBENTRY_MODEL_PARAMS = "model_parameters"

TOOL_USE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

BROWSER_SEARCH_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

CODE_INTERPRETER_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
