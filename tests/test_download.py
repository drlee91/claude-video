"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The request must stay bounded to a handful of
explicit language codes.

This fork detects the video's own spoken language and requests that track
alongside English, rather than hardcoding English-only — a German video
otherwise comes back with YouTube's auto-translated English captions and no
indication that translation happened. The bounded-request guarantee above is
unchanged: at most the detected language, its primary subtag, and ``en.*``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"

MAX_SUB_LANG_TOKENS = 3  # detected locale + primary subtag + en.*


def _capture_argv(monkeypatch: pytest.MonkeyPatch, probe_language: str = "") -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv.

    ``probe_language`` is what the language-probe call reports on stdout;
    the default of "" mimics a video with no language metadata.
    """
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = probe_language
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _sub_langs(calls: list[list[str]]) -> str:
    """Return the --sub-langs value from the one call that requests subtitles."""
    for argv in calls:
        if "--sub-langs" in argv:
            return argv[argv.index("--sub-langs") + 1]
    raise AssertionError(f"no call requested --sub-langs; calls were: {calls}")


def _assert_bounded(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert len(tokens) <= MAX_SUB_LANG_TOKENS, (
        f"sub-langs must stay a short explicit list, got {langs!r}"
    )


# --- bounded-request guard (kept from upstream) ------------------------------

def test_fetch_captions_request_stays_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls))


def test_download_url_request_stays_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls))


def test_bounded_even_when_language_detected(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, probe_language="de\n")
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls))


# --- language selection -------------------------------------------------------

def test_no_detected_language_falls_back_to_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    assert _sub_langs(calls) == "en.*"


def test_detected_language_is_requested_before_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, probe_language="de\n")
    download.fetch_captions(URL, tmp_path / "download")
    tokens = _sub_langs(calls).split(",")
    assert tokens[0] == "de", f"detected language must come first, got {tokens}"
    assert "en.*" in tokens, "English must remain available as a fallback"


def test_locale_language_also_requests_primary_subtag(monkeypatch, tmp_path):
    # YouTube keys caption tracks by primary subtag, so "en-US" alone finds
    # nothing on a video whose track is plain "en".
    calls = _capture_argv(monkeypatch, probe_language="en-US\n")
    download.fetch_captions(URL, tmp_path / "download")
    tokens = _sub_langs(calls).split(",")
    assert "en-US" in tokens and "en" in tokens, f"got {tokens}"


def test_explicit_lang_overrides_detection(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, probe_language="de\n")
    download.fetch_captions(URL, tmp_path / "download", lang="fr")
    tokens = _sub_langs(calls).split(",")
    assert tokens[0] == "fr", f"explicit --lang must win over detection, got {tokens}"
    assert "de" not in tokens


def test_explicit_lang_skips_the_probe_call(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, probe_language="de\n")
    download.fetch_captions(URL, tmp_path / "download", lang="fr")
    probes = [c for c in calls if "--print" in c]
    assert not probes, f"no language probe should run when --lang is given, got {probes}"


# --- subtitle file preference -------------------------------------------------

def test_pick_subtitle_prefers_detected_language(tmp_path):
    for name in ("video.en.vtt", "video.de.vtt"):
        (tmp_path / name).write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(tmp_path, "de")
    assert picked is not None and picked.name == "video.de.vtt"


def test_pick_subtitle_falls_back_to_english(tmp_path):
    (tmp_path / "video.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(tmp_path, "de")
    assert picked is not None and picked.name == "video.en.vtt"
