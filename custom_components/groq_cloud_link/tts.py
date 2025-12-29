import io
import wave
from functools import reduce
from typing import TYPE_CHECKING, Any

from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.components.tts.const import TtsAudioType
from homeassistant.config_entries import (
    ConfigEntry,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

if TYPE_CHECKING:
    from . import GroqDevice
from .const import DOMAIN

MAX_CHARACTERS_PER_REQUEST = 200


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry["GroqDevice"],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud Link from a config entry."""
    async_add_entities(
        hass.data[DOMAIN][entry.entry_id].entities.tts_entities(),
        update_before_add=True,
    )


class GroqTextToSpeechEntity(TextToSpeechEntity):
    """Minimal TTS entity implementation."""

    def __init__(self, device: "GroqDevice") -> None:
        """Initialize the entity."""
        super().__init__()
        self.device = device
        self._attr_unique_id = f"{self.device.settings.entry_id}_groq_tts"
        self._attr_name = "Groq Cloud TTS"
        self.default_language = "en"

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Return TTS audio data."""
        phrases: list[str] = message.split(". ")

        current_stich: list[str] = []

        audio_files: list[bytes] = []
        while True:
            current_count = reduce(
                lambda count, phrase: count + len(phrase), current_stich, 0
            )
            if (
                len(phrases) > 0
                and len(phrases[0]) + current_count < MAX_CHARACTERS_PER_REQUEST
            ):
                current_stich.append(phrases.pop(0))
            else:
                audio_files.append(
                    await (
                        await self.device.get_client().audio.speech.create(
                            model="canopylabs/orpheus-v1-english",
                            voice=self.device.settings.voice,
                            response_format="wav",
                            input=message,
                        )
                    ).read()
                )
                current_stich.clear()
                if len(phrases) == 0:
                    break

        params = None
        raw_frames = []

        for i, data in enumerate(audio_files):
            with wave.open(io.BytesIO(data), "rb") as w:
                if i == 0:
                    params = w.getparams()
                elif w.getparams() != params:
                    msg = "WAV formats must match"
                    raise ValueError(msg)

                raw_frames.append(w.readframes(w.getnframes()))

        if params is None:
            return None, None

        # Write combined WAV
        output = io.BytesIO()
        with wave.open(output, "wb") as w:
            w.setparams(params)
            for frames in raw_frames:
                w.writeframes(frames)

        return "wav", output.getvalue()
