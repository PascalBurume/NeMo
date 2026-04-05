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

"""
Extract each speaker's audio into separate files from NeMo diarization RTTM output.

This script takes an RTTM file (produced by NeMo speaker diarization) and the
original audio file, then exports one audio file per speaker containing only
that speaker's segments concatenated together.

Typical workflow:
  1. Run NeMo speaker diarization to produce an RTTM file.
  2. Run this script to split the audio by speaker.
  3. (Optional) Upload per-speaker files to a voice-cloning service.

Usage:
  python extract_speaker_audio.py \
      --audio_filepath /path/to/meeting.wav \
      --rttm_filepath /path/to/pred_rttms/meeting.rttm \
      --output_dir extracted_speakers \
      --output_format mp3

  # Process every (audio, RTTM) pair listed in a NeMo manifest:
  python extract_speaker_audio.py \
      --manifest_filepath /path/to/manifest.json \
      --rttm_dir /path/to/pred_rttms \
      --output_dir extracted_speakers \
      --output_format wav

Requirements:
  pip install soundfile numpy
  # For MP3 export: pip install pydub  (requires ffmpeg installed on the system)
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_rttm(rttm_filepath: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse an RTTM file and return a mapping from speaker ID to a sorted
    list of ``(start_seconds, end_seconds)`` tuples.

    Args:
        rttm_filepath: Path to the RTTM file.

    Returns:
        Dictionary mapping speaker labels to sorted segment lists.
    """
    speakers: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with open(rttm_filepath, "r") as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.strip().split()
            if len(parts) < 8:
                logger.warning("Skipping malformed RTTM line %d: %s", line_no, line.strip())
                continue
            if parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                logger.warning("Non-numeric timing on RTTM line %d: %s", line_no, line.strip())
                continue
            speaker = parts[7]
            speakers[speaker].append((start, start + duration))

    # Sort segments by start time for each speaker
    for speaker in speakers:
        speakers[speaker].sort()
    return dict(speakers)


def extract_speakers(
    audio_filepath: str,
    rttm_filepath: str,
    output_dir: str = "extracted_speakers",
    output_format: str = "wav",
    speaker_prefix: str = "",
) -> List[str]:
    """Extract per-speaker audio files from a diarization RTTM and source audio.

    Args:
        audio_filepath: Path to the source audio file (WAV recommended).
        rttm_filepath: Path to the RTTM file with speaker segments.
        output_dir: Directory where per-speaker files will be written.
        output_format: Output audio format (``wav`` or ``mp3``).
        speaker_prefix: Optional prefix prepended to output filenames
            (e.g. the session ID).

    Returns:
        List of paths to the exported audio files.
    """
    os.makedirs(output_dir, exist_ok=True)

    speakers = parse_rttm(rttm_filepath)
    if not speakers:
        logger.warning("No speaker segments found in %s", rttm_filepath)
        return []

    # Read audio with soundfile (supports WAV, FLAC, OGG, etc.)
    audio_data, sample_rate = sf.read(audio_filepath, dtype="float32")
    total_samples = len(audio_data)

    exported_paths: List[str] = []
    for speaker_id, segments in speakers.items():
        chunks: List[np.ndarray] = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            if start_sample >= end_sample:
                continue
            chunks.append(audio_data[start_sample:end_sample])

        if not chunks:
            logger.warning("Speaker %s has no valid audio segments — skipping.", speaker_id)
            continue

        speaker_audio = np.concatenate(chunks)
        duration_sec = len(speaker_audio) / sample_rate

        filename = f"{speaker_prefix}{speaker_id}.{output_format}" if speaker_prefix else f"{speaker_id}.{output_format}"
        output_path = os.path.join(output_dir, filename)

        if output_format == "mp3":
            _export_mp3(speaker_audio, sample_rate, output_path)
        else:
            sf.write(output_path, speaker_audio, sample_rate)

        logger.info(
            "Exported %s: %d segments, %.1fs → %s",
            speaker_id,
            len(segments),
            duration_sec,
            output_path,
        )
        exported_paths.append(output_path)

    return exported_paths


def _export_mp3(audio_data: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Export audio data to MP3 using pydub (requires ffmpeg).

    Args:
        audio_data: NumPy array of audio samples (float32, mono or multi-channel).
        sample_rate: Sample rate of the audio.
        output_path: Destination file path.
    """
    try:
        from pydub import AudioSegment  # noqa: F811
    except ImportError:
        raise ImportError(
            "pydub is required for MP3 export. Install it with: pip install pydub\n"
            "You also need ffmpeg installed on your system."
        )

    # Convert float32 [-1, 1] to int16
    audio_int16 = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    channels = 1 if audio_int16.ndim == 1 else audio_int16.shape[1]

    segment = AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,  # 16-bit
        frame_rate=sample_rate,
        channels=channels,
    )
    segment.export(output_path, format="mp3")


def process_manifest(
    manifest_filepath: str,
    rttm_dir: Optional[str],
    output_dir: str,
    output_format: str,
) -> List[str]:
    """Process every entry in a NeMo-style JSON manifest.

    Each line in the manifest is a JSON object with at least ``audio_filepath``.
    The corresponding RTTM is resolved from either the manifest's
    ``rttm_filepath`` field or by matching the audio basename inside
    *rttm_dir*.

    Args:
        manifest_filepath: Path to the JSON-lines manifest file.
        rttm_dir: Directory containing predicted RTTM files.  Used when the
            manifest does not carry ``rttm_filepath`` entries.
        output_dir: Root output directory.
        output_format: ``wav`` or ``mp3``.

    Returns:
        Flat list of all exported file paths.
    """
    all_exported: List[str] = []

    with open(manifest_filepath, "r") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON manifest line %d", line_no)
                continue

            audio_filepath = entry.get("audio_filepath")
            if not audio_filepath:
                logger.warning("Manifest line %d has no audio_filepath — skipping.", line_no)
                continue

            # Resolve RTTM path
            rttm_filepath = entry.get("rttm_filepath")
            if not rttm_filepath and rttm_dir:
                basename = os.path.splitext(os.path.basename(audio_filepath))[0]
                candidate = os.path.join(rttm_dir, f"{basename}.rttm")
                if os.path.isfile(candidate):
                    rttm_filepath = candidate

            if not rttm_filepath or not os.path.isfile(rttm_filepath):
                logger.warning(
                    "No RTTM found for %s (manifest line %d) — skipping.",
                    audio_filepath,
                    line_no,
                )
                continue

            session_id = os.path.splitext(os.path.basename(audio_filepath))[0]
            session_output_dir = os.path.join(output_dir, session_id)
            exported = extract_speakers(
                audio_filepath=audio_filepath,
                rttm_filepath=rttm_filepath,
                output_dir=session_output_dir,
                output_format=output_format,
            )
            all_exported.extend(exported)

    return all_exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-speaker audio files from NeMo diarization RTTM output."
    )

    # --- Single-file mode ---
    parser.add_argument(
        "--audio_filepath",
        type=str,
        default=None,
        help="Path to the source audio file (WAV recommended).",
    )
    parser.add_argument(
        "--rttm_filepath",
        type=str,
        default=None,
        help="Path to the RTTM file produced by NeMo diarization.",
    )

    # --- Manifest mode ---
    parser.add_argument(
        "--manifest_filepath",
        type=str,
        default=None,
        help="Path to a NeMo-style JSON-lines manifest. "
        "When provided, each entry is processed and --audio_filepath / --rttm_filepath are ignored.",
    )
    parser.add_argument(
        "--rttm_dir",
        type=str,
        default=None,
        help="Directory containing predicted RTTM files (used with --manifest_filepath).",
    )

    # --- Common options ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_speakers",
        help="Directory where per-speaker audio files will be saved (default: extracted_speakers).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=["wav", "mp3"],
        help="Output audio format: wav (default) or mp3. MP3 requires pydub + ffmpeg.",
    )

    args = parser.parse_args()

    if args.manifest_filepath:
        exported = process_manifest(
            manifest_filepath=args.manifest_filepath,
            rttm_dir=args.rttm_dir,
            output_dir=args.output_dir,
            output_format=args.output_format,
        )
    elif args.audio_filepath and args.rttm_filepath:
        exported = extract_speakers(
            audio_filepath=args.audio_filepath,
            rttm_filepath=args.rttm_filepath,
            output_dir=args.output_dir,
            output_format=args.output_format,
        )
    else:
        parser.error(
            "Provide either --manifest_filepath or both --audio_filepath and --rttm_filepath."
        )
        return

    logger.info("Done — %d speaker file(s) exported to %s", len(exported), args.output_dir)


if __name__ == "__main__":
    main()
