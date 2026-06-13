"""Whisper speech-to-text (faster-whisper) for cross-browser voice input.

The browser records mic audio (MediaRecorder, webm/opus) and POSTs it to
/api/transcribe; this transcribes it server-side so voice works even in browsers
that disable the Web Speech API (Brave, Firefox). Model is lazy-loaded once.
"""
from __future__ import annotations

import io
import os
import threading

_model = None
_lock = threading.Lock()
_load_error: str | None = None


def _get_model():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    with _lock:
        if _model is None and _load_error is None:
            try:
                from faster_whisper import WhisperModel

                size = os.environ.get("WHISPER_MODEL", "base")
                # int8 on CPU is fast and light; first load downloads the model.
                _model = WhisperModel(size, device="cpu", compute_type="int8")
            except Exception as exc:  # noqa: BLE001
                _load_error = str(exc)
    return _model


def available() -> bool:
    return _get_model() is not None


def transcribe_audio(data: bytes, language: str | None = "vi") -> str:
    """Transcribe raw audio bytes (any container ffmpeg/PyAV can decode) to text."""
    model = _get_model()
    if model is None:
        raise RuntimeError(_load_error or "Whisper model unavailable")
    segments, _info = model.transcribe(
        io.BytesIO(data),
        language=language,  # None => auto-detect
        beam_size=1,
        vad_filter=True,  # drop silence so trailing quiet doesn't add latency
    )
    return "".join(seg.text for seg in segments).strip()
