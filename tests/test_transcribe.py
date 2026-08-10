#!/usr/bin/env python3
"""Regression tests for transcribe.py's VTT dedup logic.

Run with: python3 tests/test_transcribe.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from transcribe import parse_vtt  # noqa: E402


def _parse(vtt_text: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_text)
        path = f.name
    try:
        return parse_vtt(path)
    finally:
        Path(path).unlink()


def _all_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments)


class RollingDuplicateRealSample(unittest.TestCase):
    """Actual excerpt captured from a real YouTube German auto-caption VTT
    (video nErRpN1FSrM, 2026-08-10). Reproduces the exact rolling-window
    3-cue pattern: growing content cue -> ~10ms settle transition -> next
    growing content cue whose first line repeats the settled text.
    """

    VTT = """WEBVTT
Kind: captions
Language: de

00:00:07.069 --> 00:00:09.470 align:start position:0%
[Musik]
ich<00:00:08.069><c> begrüße</c><00:00:08.429><c> ich</c><00:00:08.460><c> herzlich</c><00:00:09.000><c> zu</c><00:00:09.090><c> einer</c><00:00:09.210><c> neuen</c>

00:00:09.470 --> 00:00:09.480 align:start position:0%
ich begrüße ich herzlich zu einer neuen


00:00:09.480 --> 00:00:11.810 align:start position:0%
ich begrüße ich herzlich zu einer neuen
folge<00:00:09.840><c> von</c><00:00:10.050><c> alpha</c><00:00:10.710><c> baby</c><00:00:11.099><c> mein</c><00:00:11.400><c> name</c><00:00:11.670><c> ist</c>

00:00:11.810 --> 00:00:11.820 align:start position:0%
folge von alpha baby mein name ist


00:00:11.820 --> 00:00:13.910 align:start position:0%
folge von alpha baby mein name ist
antrag<00:00:12.210><c> fabry</c><00:00:12.570><c> und</c><00:00:12.809><c> als</c><00:00:13.349><c> astrologin</c>
"""

    def test_no_line_appears_twice_in_output(self):
        segments = _parse(self.VTT)
        combined = _all_text(segments)
        # Each rolling-window line must appear exactly once in the reconstructed text.
        for phrase in [
            "ich begrüße ich herzlich zu einer neuen",
            "folge von alpha baby mein name ist",
        ]:
            self.assertEqual(
                combined.count(phrase), 1,
                f"expected {phrase!r} exactly once, found {combined.count(phrase)} in: {combined!r}",
            )

    def test_reconstructs_full_sentence_without_gaps_or_dupes(self):
        segments = _parse(self.VTT)
        combined = _all_text(segments)
        self.assertEqual(
            combined,
            "[Musik] ich begrüße ich herzlich zu einer neuen "
            "folge von alpha baby mein name ist antrag fabry und als astrologin",
        )

    def test_segment_timestamps_stay_monotonic_and_in_range(self):
        segments = _parse(self.VTT)
        self.assertGreater(len(segments), 0)
        prev_end = -1.0
        for seg in segments:
            self.assertGreaterEqual(seg["start"], prev_end)
            self.assertGreaterEqual(seg["end"], seg["start"])
            prev_end = seg["end"]
        self.assertGreaterEqual(segments[0]["start"], 7.0)
        self.assertLessEqual(segments[-1]["end"], 13.91)


class ExactDuplicateCueStillCollapses(unittest.TestCase):
    """The simple case the old code already handled must keep working."""

    VTT = """WEBVTT

00:00:01.000 --> 00:00:02.000
Hallo zusammen

00:00:02.000 --> 00:00:03.000
Hallo zusammen

00:00:03.000 --> 00:00:05.000
und willkommen
"""

    def test_identical_consecutive_lines_collapse_once(self):
        segments = _parse(self.VTT)
        combined = _all_text(segments)
        self.assertEqual(combined.count("Hallo zusammen"), 1)
        self.assertIn("und willkommen", combined)


class GenuineRepetitionIsPreserved(unittest.TestCase):
    """Safety case flagged during review: a real spoken repetition (e.g. an
    idiom said twice) must NOT be treated as a rolling-caption artifact just
    because it shares words with the previous line. Only a fully identical
    line is ever collapsed — a partial/substring overlap is not enough.
    """

    VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
das war das erste Mal dass ich sowas erlebt habe

00:00:03.000 --> 00:00:05.000
das erste Mal war ich total überrascht
"""

    def test_repeated_idiom_across_distinct_lines_not_dropped(self):
        segments = _parse(self.VTT)
        combined = _all_text(segments)
        self.assertEqual(combined.count("das erste Mal"), 2)
        self.assertIn("das war das erste Mal dass ich sowas erlebt habe", combined)
        self.assertIn("das erste Mal war ich total überrascht", combined)


class SegmentGrouping(unittest.TestCase):
    def test_long_silence_starts_a_new_segment(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:02.000
erste Zeile

00:00:10.000 --> 00:00:11.000
zweite Zeile nach einer Pause
"""
        segments = _parse(vtt)
        self.assertEqual(len(segments), 2)

    def test_long_continuous_speech_is_chunked(self):
        lines = []
        t = 0.0
        for i in range(40):
            start, end = t, t + 0.5
            lines.append(f"00:00:{start:06.3f} --> 00:00:{end:06.3f}\nwort{i}\n")
            t = end
        vtt = "WEBVTT\n\n" + "\n".join(lines)
        segments = _parse(vtt)
        self.assertGreater(len(segments), 1, "40 words with no repeats should split into multiple segments")
        tokens = _all_text(segments).split()
        for i in range(40):
            self.assertEqual(tokens.count(f"wort{i}"), 1)


if __name__ == "__main__":
    unittest.main()
