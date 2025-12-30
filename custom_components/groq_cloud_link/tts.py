import io
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile
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

    @staticmethod
    def __stich(audio_files: list[io.BytesIO]) -> bytes:
        sample_rate: int | None = None
        stiched: np.ndarray | None = None
        for audio_file in audio_files:
            data: np.ndarray
            samplerate: int
            data, samplerate = soundfile.read(audio_file, dtype="int16")
            if sample_rate is None or stiched is None:
                sample_rate = samplerate
                stiched = data
            elif sample_rate != samplerate:
                msg = f"Expected sample rate of {sample_rate}, got {samplerate}"
                raise Exception(msg)
            np.append(stiched, data)

        buffer = io.BytesIO()
        soundfile.write(buffer, stiched, sample_rate, format="WAV")

        return buffer.getvalue()

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Return TTS audio data."""
        phrases: list[str] = message.split(". ")

        current_stich: list[str] = []

        audio_files: list[io.BytesIO] = []
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
                    io.BytesIO(
                        await (
                            await self.device.get_client().audio.speech.create(
                                model="canopylabs/orpheus-v1-english",
                                voice=self.device.settings.voice,
                                response_format="wav",
                                input=". ".join(current_stich),
                            )
                        ).read()
                    )
                )
                current_stich.clear()
                if len(phrases) == 0:
                    break

        return "wav", GroqTextToSpeechEntity.__stich(audio_files)
