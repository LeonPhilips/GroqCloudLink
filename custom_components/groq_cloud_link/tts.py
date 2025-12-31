import io
import re
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
    def __split_long_sentence(sentence: str, max_len: int) -> list[str]:
        words = sentence.split(" ")
        chunks = []
        current = ""
        for word in words:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_len:
                current += " " + word
            else:
                chunks.append(current)
                current = word
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def __split_into_blocks(text: str, max_len: int) -> list[str]:
        sentences = re.findall(r"[^.!?]+[.!?]", text)
        blocks = []
        current = ""
        for sentence in (s.strip() for s in sentences):
            if len(sentence) > max_len:
                if current:
                    blocks.append(current)
                    current = ""
                blocks.extend(GroqTextToSpeechEntity.__split_long_sentence(sentence, max_len))
                continue

            if not current:
                current = sentence
            elif len(current) + len(sentence) < max_len:
                current += " " + sentence
            else:
                blocks.append(current)
                current = sentence
        if current:
            blocks.append(current)
        return blocks

    @staticmethod
    def __stich(audio_files: list[io.BytesIO]) -> bytes | None:
        if len(audio_files) == 0:
            return None

        sample_rate: int | None = None
        stiched: list[np.ndarray] = []
        for audio_file in audio_files:
            data: np.ndarray
            samplerate: int
            data, samplerate = soundfile.read(audio_file, dtype="int16")
            if sample_rate is None:
                sample_rate = samplerate
            elif sample_rate != samplerate:
                msg = f"Expected sample rate of {sample_rate}, got {samplerate}"
                raise Exception(msg)
            stiched.append(data)

        buffer = io.BytesIO()
        soundfile.write(buffer, np.concatenate(stiched), sample_rate, format="FLAC")

        return buffer.getvalue()

    async def async_get_tts_audio(self, message: str, language: str, options: dict[str, Any]) -> TtsAudioType:
        """Return TTS audio data."""
        phrases: list[str] = GroqTextToSpeechEntity.__split_into_blocks(message, MAX_CHARACTERS_PER_REQUEST)
        audio_files: list[io.BytesIO] = []

        for phrase in phrases:
            audio_files.append(  # noqa: PERF401 Code readability > micro performance improvements
                io.BytesIO(
                    await (
                        await self.device.get_client().audio.speech.create(
                            model="canopylabs/orpheus-v1-english",
                            voice=self.device.settings.voice,
                            response_format="wav",
                            input=phrase,
                        )
                    ).read()
                )
            )

        return "flac", GroqTextToSpeechEntity.__stich(audio_files)
