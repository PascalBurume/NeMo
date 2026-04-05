# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for scripts/speaker_tasks/extract_speaker_audio.py."""

import json
import os

import numpy as np
import soundfile as sf

# Import the module under test
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "speaker_tasks"))
from extract_speaker_audio import extract_speakers, parse_rttm, process_manifest


def _make_rttm(path: str, lines: list) -> None:
    """Write RTTM lines to *path*."""
    with open(path, "w") as fh:
        for line in lines:
            fh.write(line + "\n")


def _make_wav(path: str, duration_sec: float = 10.0, sample_rate: int = 16000) -> None:
    """Create a mono WAV file filled with a sine wave."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sample_rate)


class TestParseRttm:
    """Tests for parse_rttm()."""

    def test_basic_parse(self, tmp_path):
        rttm = str(tmp_path / "test.rttm")
        _make_rttm(
            rttm,
            [
                "SPEAKER file1 1 0.5 2.0 <NA> <NA> speaker_0 <NA> <NA>",
                "SPEAKER file1 1 3.0 1.5 <NA> <NA> speaker_1 <NA> <NA>",
                "SPEAKER file1 1 5.0 2.0 <NA> <NA> speaker_0 <NA> <NA>",
            ],
        )
        speakers = parse_rttm(rttm)
        assert set(speakers.keys()) == {"speaker_0", "speaker_1"}
        assert speakers["speaker_0"] == [(0.5, 2.5), (5.0, 7.0)]
        assert speakers["speaker_1"] == [(3.0, 4.5)]

    def test_empty_rttm(self, tmp_path):
        rttm = str(tmp_path / "empty.rttm")
        _make_rttm(rttm, [])
        speakers = parse_rttm(rttm)
        assert speakers == {}

    def test_malformed_lines_skipped(self, tmp_path):
        rttm = str(tmp_path / "bad.rttm")
        _make_rttm(
            rttm,
            [
                "SPEAKER file1 1 0.5 2.0 <NA> <NA> speaker_0 <NA> <NA>",
                "SHORT LINE",
                "SPEAKER file1 1 bad_float 2.0 <NA> <NA> speaker_1 <NA> <NA>",
            ],
        )
        speakers = parse_rttm(rttm)
        # Only the first valid line should be parsed
        assert set(speakers.keys()) == {"speaker_0"}

    def test_segments_sorted_by_start(self, tmp_path):
        rttm = str(tmp_path / "unsorted.rttm")
        _make_rttm(
            rttm,
            [
                "SPEAKER f 1 5.0 1.0 <NA> <NA> spk <NA> <NA>",
                "SPEAKER f 1 1.0 1.0 <NA> <NA> spk <NA> <NA>",
                "SPEAKER f 1 3.0 1.0 <NA> <NA> spk <NA> <NA>",
            ],
        )
        speakers = parse_rttm(rttm)
        starts = [s for s, _ in speakers["spk"]]
        assert starts == [1.0, 3.0, 5.0]


class TestExtractSpeakers:
    """Tests for extract_speakers()."""

    def test_extract_wav(self, tmp_path):
        wav_path = str(tmp_path / "audio.wav")
        rttm_path = str(tmp_path / "audio.rttm")
        out_dir = str(tmp_path / "out")

        _make_wav(wav_path, duration_sec=10.0, sample_rate=16000)
        _make_rttm(
            rttm_path,
            [
                "SPEAKER audio 1 0.0 2.0 <NA> <NA> speaker_0 <NA> <NA>",
                "SPEAKER audio 1 3.0 2.0 <NA> <NA> speaker_1 <NA> <NA>",
                "SPEAKER audio 1 6.0 1.0 <NA> <NA> speaker_0 <NA> <NA>",
            ],
        )

        exported = extract_speakers(wav_path, rttm_path, output_dir=out_dir, output_format="wav")

        assert len(exported) == 2
        assert os.path.isfile(os.path.join(out_dir, "speaker_0.wav"))
        assert os.path.isfile(os.path.join(out_dir, "speaker_1.wav"))

        # Verify durations
        data0, sr0 = sf.read(os.path.join(out_dir, "speaker_0.wav"))
        data1, sr1 = sf.read(os.path.join(out_dir, "speaker_1.wav"))
        assert sr0 == 16000
        # speaker_0 has 2.0s + 1.0s = 3.0s
        assert abs(len(data0) / sr0 - 3.0) < 0.01
        # speaker_1 has 2.0s
        assert abs(len(data1) / sr1 - 2.0) < 0.01

    def test_empty_rttm_returns_empty(self, tmp_path):
        wav_path = str(tmp_path / "audio.wav")
        rttm_path = str(tmp_path / "empty.rttm")
        _make_wav(wav_path, duration_sec=5.0)
        _make_rttm(rttm_path, [])

        exported = extract_speakers(wav_path, rttm_path, output_dir=str(tmp_path / "out"))
        assert exported == []

    def test_segment_beyond_audio_end_is_clamped(self, tmp_path):
        wav_path = str(tmp_path / "short.wav")
        rttm_path = str(tmp_path / "short.rttm")
        out_dir = str(tmp_path / "out")

        _make_wav(wav_path, duration_sec=3.0, sample_rate=16000)
        _make_rttm(
            rttm_path,
            [
                # Segment extends past the 3-second audio
                "SPEAKER f 1 2.0 5.0 <NA> <NA> speaker_0 <NA> <NA>",
            ],
        )

        exported = extract_speakers(wav_path, rttm_path, output_dir=out_dir, output_format="wav")
        assert len(exported) == 1
        data, sr = sf.read(os.path.join(out_dir, "speaker_0.wav"))
        # Should be clamped to ~1.0s (from 2.0 to 3.0)
        assert abs(len(data) / sr - 1.0) < 0.01

    def test_speaker_prefix(self, tmp_path):
        wav_path = str(tmp_path / "audio.wav")
        rttm_path = str(tmp_path / "audio.rttm")
        out_dir = str(tmp_path / "out")

        _make_wav(wav_path, duration_sec=5.0)
        _make_rttm(
            rttm_path,
            ["SPEAKER f 1 0.0 2.0 <NA> <NA> speaker_0 <NA> <NA>"],
        )

        exported = extract_speakers(
            wav_path, rttm_path, output_dir=out_dir, output_format="wav", speaker_prefix="session1_"
        )
        assert len(exported) == 1
        assert "session1_speaker_0.wav" in exported[0]


class TestProcessManifest:
    """Tests for process_manifest()."""

    def test_manifest_mode(self, tmp_path):
        # Create audio + rttm
        wav_path = str(tmp_path / "meeting.wav")
        rttm_path = str(tmp_path / "rttms" / "meeting.rttm")
        os.makedirs(os.path.dirname(rttm_path), exist_ok=True)

        _make_wav(wav_path, duration_sec=8.0)
        _make_rttm(
            rttm_path,
            [
                "SPEAKER meeting 1 0.0 3.0 <NA> <NA> speaker_0 <NA> <NA>",
                "SPEAKER meeting 1 4.0 2.0 <NA> <NA> speaker_1 <NA> <NA>",
            ],
        )

        # Create manifest
        manifest_path = str(tmp_path / "manifest.json")
        with open(manifest_path, "w") as fh:
            fh.write(json.dumps({"audio_filepath": wav_path, "offset": 0, "duration": None}) + "\n")

        out_dir = str(tmp_path / "output")
        exported = process_manifest(
            manifest_filepath=manifest_path,
            rttm_dir=str(tmp_path / "rttms"),
            output_dir=out_dir,
            output_format="wav",
        )

        assert len(exported) == 2
        assert os.path.isdir(os.path.join(out_dir, "meeting"))

    def test_manifest_with_rttm_in_entry(self, tmp_path):
        wav_path = str(tmp_path / "call.wav")
        rttm_path = str(tmp_path / "call.rttm")

        _make_wav(wav_path, duration_sec=5.0)
        _make_rttm(
            rttm_path,
            ["SPEAKER call 1 0.0 2.0 <NA> <NA> speaker_0 <NA> <NA>"],
        )

        manifest_path = str(tmp_path / "manifest.json")
        with open(manifest_path, "w") as fh:
            entry = {"audio_filepath": wav_path, "rttm_filepath": rttm_path}
            fh.write(json.dumps(entry) + "\n")

        out_dir = str(tmp_path / "output")
        exported = process_manifest(
            manifest_filepath=manifest_path,
            rttm_dir=None,
            output_dir=out_dir,
            output_format="wav",
        )
        assert len(exported) == 1
