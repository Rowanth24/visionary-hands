"""Speech output for the live pipeline — abstracted so Jetson can swap engines later.

SayTTS shells out to macOS `say`; Pyttsx3TTS drives pyttsx3 if installed (the
espeak path on Linux/Jetson); NullTTS is the silent fallback that keeps the
pipeline runnable with no audio stack. create_tts() picks the first available.

All engines share a debounce: identical text is never re-spoken within the
window (default 3s), so a sliding-window recognizer emitting the same word
every stride does not stutter the speaker.
"""
from __future__ import annotations

import shutil
import subprocess
import time

DEFAULT_DEBOUNCE_SECONDS = 3.0


class BaseTTS:
    name = "base"

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS):
        self.debounce_seconds = float(debounce_seconds)
        self._last_text: str | None = None
        self._last_time: float | None = None

    def speak(self, text) -> bool:
        """Speak text unless the identical text was spoken within the debounce window.

        Returns True if spoken, False if suppressed (debounce) or text is empty.
        """
        text = str(text).strip()
        if not text:
            return False
        now = time.monotonic()
        if (
            text == self._last_text
            and self._last_time is not None
            and (now - self._last_time) < self.debounce_seconds
        ):
            return False
        self._speak(text)
        self._last_text, self._last_time = text, now
        return True

    def _speak(self, text: str) -> None:
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        return True


class SayTTS(BaseTTS):
    """macOS built-in `say` command (blocking; utterances are single short words)."""

    name = "say"

    def _speak(self, text: str) -> None:
        subprocess.run(["say", text], check=False)

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("say") is not None


class Pyttsx3TTS(BaseTTS):
    """Cross-platform pyttsx3 engine (optional dependency; espeak on Linux)."""

    name = "pyttsx3"

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS):
        super().__init__(debounce_seconds)
        import pyttsx3

        self._engine = pyttsx3.init()

    def _speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            return False
        return True


class NullTTS(BaseTTS):
    """Silent engine — same contract, no side effects."""

    name = "null"

    def _speak(self, text: str) -> None:
        pass


_ENGINES = {"say": SayTTS, "pyttsx3": Pyttsx3TTS, "null": NullTTS}
_DEFAULT_ORDER = ["say", "pyttsx3", "null"]


def create_tts(prefer: str | None = None,
               debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS) -> BaseTTS:
    """Return the first available engine, trying `prefer` first (say -> pyttsx3 -> null)."""
    order = list(_DEFAULT_ORDER)
    if prefer is not None:
        if prefer not in _ENGINES:
            raise ValueError(f"Unknown TTS engine: {prefer!r} (choose from {sorted(_ENGINES)})")
        order = [prefer] + [name for name in order if name != prefer]
    for name in order:
        engine_cls = _ENGINES[name]
        if engine_cls.is_available():
            return engine_cls(debounce_seconds=debounce_seconds)
    return NullTTS(debounce_seconds=debounce_seconds)
