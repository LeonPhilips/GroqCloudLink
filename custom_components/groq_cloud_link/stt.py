import io
import wave
from asyncio import CancelledError
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING

import groq
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import (
    ConfigEntry,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

if TYPE_CHECKING:
    from . import GroqDevice
from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry["GroqDevice"],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Cloud Link from a config entry."""
    async_add_entities(
        hass.data[DOMAIN][entry.entry_id].entities.stt_entities(),
        update_before_add=True,
    )


class GroqSpeechToTextEntity(SpeechToTextEntity):
    _attr_name = "Groq Cloud STT"

    def __init__(self, device: "GroqDevice") -> None:
        super().__init__()
        self.device = device
        self._attr_unique_id = f"{self.device.settings.entry_id}_groq_stt"

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en-US",
            "zh-CN",
            "de-DE",
            "es-ES",
            "ru-RU",
            "ko-KR",
            "fr-FR",
            "ja-JP",
            "pt-BR",
            "tr-TR",
            "pl-PL",
            "ca-ES",
            "nl-NL",
            "ar-SA",
            "sv-SE",
            "it-IT",
            "id-ID",
            "hi-IN",
            "fi-FI",
            "vi-VN",
            "he-IL",
            "uk-UA",
            "el-GR",
            "ms-MY",
            "cs-CZ",
            "ro-RO",
            "da-DK",
            "hu-HU",
            "ta-IN",
            "no-NO",
            "th-TH",
            "ur-PK",
            "hr-HR",
            "bg-BG",
            "lt-LT",
            "la-VA",
            "mi-NZ",
            "ml-IN",
            "cy-GB",
            "sk-SK",
            "te-IN",
            "fa-IR",
            "lv-LV",
            "bn-IN",
            "sr-RS",
            "az-AZ",
            "sl-SI",
            "kn-IN",
            "et-EE",
            "mk-MK",
            "br-FR",
            "eu-ES",
            "is-IS",
            "hy-AM",
            "ne-NP",
            "mn-MN",
            "bs-BA",
            "kk-KZ",
            "sq-AL",
            "sw-KE",
            "gl-ES",
            "mr-IN",
            "pa-IN",
            "si-LK",
            "km-KH",
            "sn-ZW",
            "yo-NG",
            "so-SO",
            "af-ZA",
            "oc-FR",
            "ka-GE",
            "be-BY",
            "tg-TJ",
            "sd-PK",
            "gu-IN",
            "am-ET",
            "lo-LA",
            "uz-UZ",
            "fo-FO",
            "ht-HT",
            "ps-AF",
            "tk-TM",
            "nn-NO",
            "mt-MT",
            "sa-IN",
            "lb-LU",
            "my-MM",
            "bo-CN",
            "tl-PH",
            "mg-MG",
            "as-IN",
            "tt-RU",
            "ln-CD",
            "ha-NG",
            "ba-RU",
            "jv-ID",
            "su-ID",
        ]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]) -> SpeechResult:
        try:
            pcm_audio = b""
            async for chunk in stream:
                pcm_audio += chunk

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm_audio)
            wav_buffer.seek(0)

            transcription = await self.device.get_client().audio.transcriptions.create(
                file=("input.wav", wav_buffer),
                model=self.device.settings.stt_model,
                temperature=self.device.settings.stt_temperature,
                response_format="verbose_json",
                language=metadata.language,
            )
            return SpeechResult(transcription.text, SpeechResultState.SUCCESS)
        except groq.APIError as e:
            return SpeechResult(e.message, SpeechResultState.ERROR)
        except CancelledError:
            return SpeechResult(None, SpeechResultState.ERROR)
        except BaseException:
            return SpeechResult(None, SpeechResultState.ERROR)
