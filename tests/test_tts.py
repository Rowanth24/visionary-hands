#!/usr/bin/env python3
"""Tests for pipeline/tts.py — engine selection + debounce (Gate W7, no audio in CI).

Run:  .venv-oracle/bin/pytest tests/test_tts.py -q   (after `source env.sh`)
"""
from __future__ import annotations

import sys
import types

import pipeline.tts as tts_mod
from pipeline.tts import NullTTS, Pyttsx3TTS, SayTTS, create_tts


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now


# --- SayTTS: shells out to macOS `say`, nothing else ---

def test_say_tts_invokes_say_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert SayTTS().speak("bowling") is True
    assert calls == [["say", "bowling"]]


def test_empty_text_is_not_spoken(monkeypatch):
    calls = []
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert SayTTS().speak("   ") is False
    assert calls == []


# --- debounce: identical text suppressed within the window, allowed after ---

def test_debounce_suppresses_identical_text_within_window(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(tts_mod.time, "monotonic", clock)
    calls = []
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    tts = SayTTS(debounce_seconds=3.0)

    assert tts.speak("drink") is True
    clock.now = 1.0
    assert tts.speak("drink") is False          # repeat inside window -> suppressed
    assert len(calls) == 1


def test_debounce_allows_identical_text_after_window(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(tts_mod.time, "monotonic", clock)
    calls = []
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    tts = SayTTS(debounce_seconds=3.0)

    tts.speak("drink")
    clock.now = 3.5
    assert tts.speak("drink") is True           # window elapsed -> spoken again
    assert len(calls) == 2


def test_debounce_does_not_block_different_text(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(tts_mod.time, "monotonic", clock)
    calls = []
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    tts = SayTTS(debounce_seconds=3.0)

    tts.speak("drink")
    clock.now = 0.5
    assert tts.speak("yes") is True             # different text -> never debounced
    assert calls == [["say", "drink"], ["say", "yes"]]


# --- NullTTS: honors the same contract, produces no side effects ---

def test_null_tts_speaks_silently():
    tts = NullTTS()
    assert tts.speak("bowling") is True
    assert tts.speak("bowling") is False        # debounce applies to Null too


# --- Pyttsx3TTS: optional import, engine driven via say/runAndWait ---

def _fake_pyttsx3(monkeypatch):
    engine = types.SimpleNamespace(spoken=[])
    engine.say = lambda text: engine.spoken.append(text)
    engine.runAndWait = lambda: None
    module = types.SimpleNamespace(init=lambda: engine)
    monkeypatch.setitem(sys.modules, "pyttsx3", module)
    return engine


def test_pyttsx3_tts_uses_engine(monkeypatch):
    engine = _fake_pyttsx3(monkeypatch)
    tts = Pyttsx3TTS()
    assert tts.speak("deaf") is True
    assert engine.spoken == ["deaf"]


def test_pyttsx3_unavailable_without_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyttsx3", None)   # import -> ImportError
    assert Pyttsx3TTS.is_available() is False


# --- create_tts: first available engine, prefer honored ---

def test_create_tts_prefers_say_when_available(monkeypatch):
    monkeypatch.setattr(tts_mod.shutil, "which", lambda name: "/usr/bin/say")
    assert isinstance(create_tts(), SayTTS)


def test_create_tts_falls_back_to_pyttsx3_then_null(monkeypatch):
    monkeypatch.setattr(tts_mod.shutil, "which", lambda name: None)   # no `say`
    _fake_pyttsx3(monkeypatch)
    assert isinstance(create_tts(), Pyttsx3TTS)

    monkeypatch.setitem(sys.modules, "pyttsx3", None)                 # no pyttsx3 either
    assert isinstance(create_tts(), NullTTS)


def test_create_tts_prefer_null_skips_available_say(monkeypatch):
    monkeypatch.setattr(tts_mod.shutil, "which", lambda name: "/usr/bin/say")
    assert isinstance(create_tts(prefer="null"), NullTTS)


def test_create_tts_rejects_unknown_engine():
    try:
        create_tts(prefer="espeak-someday")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown engine name")
