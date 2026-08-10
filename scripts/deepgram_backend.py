#!/usr/bin/env python3
"""Transcribe an audio file via the Deepgram pre-recorded API.

Used as a large-file fallback for the Whisper path: Groq and OpenAI both cap
uploads at 25 MB, so audio from long videos (~45 min+ at 64 kbps) gets rejected
with HTTP 413. Deepgram's pre-recorded endpoint accepts files up to ~2 GB and
returns utterance-level timestamps, which map directly onto our {start, end,
text} segment shape.

Pure stdlib — no `pip install deepgram-sdk` / `httpx`. Named `deepgram_backend`
(not `deepgram`) so it never shadows or gets shadowed by the real Deepgram SDK
if that happens to be installed in the same environment.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen


DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-2"

# Query options. nova-2 with detect_language keeps this language-agnostic (the
# /watch skill is general-purpose, not German-only). utterances gives clean
# timestamped chunks; smart_format + punctuate make the text readable.
DEEPGRAM_PARAMS = {
    "model": DEEPGRAM_MODEL,
    "smart_format": "true",
    "punctuate": "true",
    "utterances": "true",
    "detect_language": "true",
}


def load_deepgram_key() -> str | None:
    """Read DEEPGRAM_API_KEY from the environment or the watch .env files."""
    value = os.environ.get("DEEPGRAM_API_KEY")
    if value and value.strip():
        return value.strip()

    dotenv_paths = [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]
    for path in dotenv_paths:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                if key.strip() != "DEEPGRAM_API_KEY":
                    continue
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                    raw = raw[1:-1]
                if raw:
                    return raw
        except OSError:
            continue
    return None


def _mimetype(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")


def _segments_from_response(data: dict) -> list[dict]:
    """Convert a Deepgram response into our {start, end, text} segment list.

    Prefer utterances (already chunked with timestamps); fall back to
    paragraph sentences, then to the flat transcript with a single segment.
    """
    results = (data or {}).get("results") or {}

    out: list[dict] = []
    for utt in results.get("utterances") or []:
        text = str(utt.get("transcript") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(utt.get("start") or 0.0), 2),
            "end": round(float(utt.get("end") or 0.0), 2),
            "text": text,
        })
    if out:
        return out

    channels = results.get("channels") or []
    alts = (channels[0].get("alternatives") if channels else None) or []
    alt = alts[0] if alts else {}

    paragraphs = ((alt.get("paragraphs") or {}).get("paragraphs")) or []
    for para in paragraphs:
        sentences = para.get("sentences") or []
        text = " ".join(str(s.get("text") or "").strip() for s in sentences).strip()
        if not text:
            continue
        out.append({
            "start": round(float(para.get("start") or 0.0), 2),
            "end": round(float(para.get("end") or 0.0), 2),
            "text": text,
        })
    if out:
        return out

    full = str(alt.get("transcript") or "").strip()
    if full:
        out.append({"start": 0.0, "end": 0.0, "text": full})
    return out


def transcribe_deepgram(audio_path: Path, api_key: str) -> list[dict]:
    """Upload audio to Deepgram and return {start, end, text} segments.

    Raises SystemExit on any hard failure so the caller can fall through or
    report frames-only, matching the Whisper path's contract.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise SystemExit(f"Deepgram: audio file missing or empty: {audio_path}")

    url = f"{DEEPGRAM_ENDPOINT}?{urllib.parse.urlencode(DEEPGRAM_PARAMS)}"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": _mimetype(audio_path),
    }
    body = audio_path.read_bytes()
    context = ssl.create_default_context()

    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=600, context=context) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = f" — {exc.read().decode('utf-8', errors='replace')[:400]}"
        except Exception:
            pass
        raise SystemExit(f"Deepgram request failed: {exc}{detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"Deepgram network error: {type(exc).__name__}: {exc}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Deepgram returned non-JSON response: {exc}: {payload[:200]}")

    segments = _segments_from_response(data)
    if not segments:
        raise SystemExit("Deepgram returned no transcript segments")
    return segments


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: deepgram_backend.py <audio-path>", file=sys.stderr)
        raise SystemExit(2)

    key = load_deepgram_key()
    if not key:
        print("No DEEPGRAM_API_KEY found (env or ~/.config/watch/.env)", file=sys.stderr)
        raise SystemExit(3)

    segs = transcribe_deepgram(Path(sys.argv[1]), key)
    print(json.dumps({"backend": "deepgram", "segments": segs}, indent=2, ensure_ascii=False))
