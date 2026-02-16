#!/usr/bin/env python3
"""
Qwen3-Next-80B Four-Pass Post-Processing for Transcription Correction
──────────────────────────────────────────────────────────────────────
Four-pass contextual correction using Qwen3-Next-80B (80B parameter model).

Pass 1: Speaker diarization (SPEAKER_XX -> actual role)
Pass 2: Transcription error correction
Pass 3: Speaker diarization refinement
Pass 4: Transcription error correction refinement

Features:
• Four-pass processing for better diarization and correction results
• Smart resume: skips already processed files
• Async processing with configurable concurrency
• Domain-specific prompts (neurochirurgie, prevention_suicide)
• Chunking at 500 segments for large files
• Benchmark JSON output for papers
• Uses OpenAI-compatible API endpoint

Usage:
    # Normal processing
    python src/post_processing/postprocess_qwen_80b_fourpass.py

    # Force reprocessing
    export POSTPROCESS_FORCE=1
    python src/post_processing/postprocess_qwen_80b_fourpass.py
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Final, List, Tuple
from openai import AsyncOpenAI

# Import centralized prompts
from domain_adapted_prompts import DIARIZATION_PROMPTS, CORRECTION_PROMPTS
from benchmark_utils import BenchmarkTimer

###############################################################################
# Configuration
###############################################################################
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_ROOT: Final[Path] = PROJECT_ROOT / "results" / "transcriptions" / "whisperx_pyannote"
OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_fourpass"

# Qwen 80B API configuration (Instruct model)
MODEL: Final[str] = "Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
BASE_URL: Final[str] = "http://localhost:8000/v1"
API_KEY: Final[str] = "your-api-key"

TEMPERATURE: Final[float] = float(os.getenv("POSTPROCESS_TEMPERATURE", "0"))
MAX_RETRIES: Final[int] = int(os.getenv("POSTPROCESS_RETRIES", "3"))
RETRY_DELAY: Final[int] = 2
CONCURRENCY: Final[int] = int(os.getenv("POSTPROCESS_CONCURRENCY", "2"))
FILE_CONCURRENCY: Final[int] = int(os.getenv("POSTPROCESS_FILE_CONCURRENCY", "1"))
FORCE: Final[bool] = os.getenv("POSTPROCESS_FORCE", "0") == "1"

# Chunk size for large files (uniform across all scripts)
CHUNK_SIZE: Final[int] = int(os.getenv("POSTPROCESS_CHUNK_SIZE", "500"))

###############################################################################
# OpenAI-compatible Client for Qwen
###############################################################################
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=900.0  # 15 minutes timeout for large models
)

###############################################################################
# Utils: parsing + LLM calls
###############################################################################


def parse_segments(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse text and extract all segments with their timestamps.

    Returns:
        List of tuples (timestamp, speaker_label, content)
    """
    segments = []
    pattern = re.compile(r'^(\d+\.?\d*\s*-\s*\d+\.?\d*)\s+(.+)$')

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        match = pattern.match(line)

        if match:
            timestamp = match.group(1)
            speaker_label = match.group(2)

            # Get content (next line)
            content = ""
            if i + 1 < len(lines) and not pattern.match(lines[i + 1]):
                content = lines[i + 1]
                i += 2
            else:
                i += 1

            segments.append((timestamp, speaker_label, content))
        else:
            i += 1

    return segments


def segments_to_text(segments: List[Tuple[str, str, str]]) -> str:
    """Convert segments back to text format."""
    output_lines = []
    for ts, spk, txt in segments:
        output_lines.append(f"{ts} {spk}")
        if txt:
            output_lines.append(txt)
        output_lines.append("")
    return '\n'.join(output_lines).strip()


def count_segments(text: str) -> int:
    """Count number of speaker segments (lines with timestamps)."""
    pattern = re.compile(r'^\d+\.?\d*\s*-\s*\d+\.?\d*\s+', re.MULTILINE)
    return len(pattern.findall(text))


async def call_qwen_text(prompt: str, segments_input: List[Tuple[str, str, str]], sem: asyncio.Semaphore) -> str:
    """
    Call Qwen API with text-based output, with strict segment count validation.

    Args:
        prompt: System prompt
        segments_input: Input segments as (timestamp, speaker, text) tuples
        sem: Semaphore for concurrency control

    Returns:
        Raw text response
    """
    # Build user message with numbered segments to enforce count
    user_message = f"Voici la transcription à traiter ({len(segments_input)} segments).\n"
    user_message += f"IMPORTANT: Tu DOIS retourner EXACTEMENT {len(segments_input)} segments dans le même format.\n\n"

    for i, (timestamp, speaker, text) in enumerate(segments_input, 1):
        user_message += f"Segment {i}/{len(segments_input)}:\n"
        user_message += f"{timestamp} {speaker}\n"
        if text:
            user_message += f"{text}\n"
        user_message += "\n"

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                completion = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt + f"\n\nCRITICAL: Retourne EXACTEMENT {len(segments_input)} segments. Ne fusionne pas, ne divise pas les segments."},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=TEMPERATURE,
                )

                result = completion.choices[0].message.content

                # Verify segment count immediately
                result_segments = parse_segments(result)
                diff = abs(len(result_segments) - len(segments_input))

                if diff > 0:
                    # Allow tolerance of up to 10% or 2 segments, whichever is larger
                    max_tolerance = max(2, int(len(segments_input) * 0.10))

                    if diff <= max_tolerance:
                        print(f"  Segment count mismatch within tolerance (got {len(result_segments)}, expected {len(segments_input)}, diff={diff})")
                        # Truncate or pad to match expected count
                        if len(result_segments) > len(segments_input):
                            result_segments = result_segments[:len(segments_input)]
                        elif len(result_segments) < len(segments_input):
                            for idx in range(len(result_segments), len(segments_input)):
                                result_segments.append(segments_input[idx])
                        return segments_to_text(result_segments)
                    elif attempt < MAX_RETRIES:
                        print(f"  Segment count mismatch (got {len(result_segments)}, expected {len(segments_input)}, diff={diff}), retrying...")
                        await asyncio.sleep(RETRY_DELAY)
                        continue

                return result

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Qwen API error after {MAX_RETRIES} retries: {exc}") from exc
                print(f"  WARNING: Retry {attempt}/{MAX_RETRIES} after error: {exc}")
                await asyncio.sleep(RETRY_DELAY * attempt)


async def process_chunks(
    prompt: str,
    segments: List[Tuple[str, str, str]],
    sem: asyncio.Semaphore
) -> List[Tuple[str, str, str]]:
    """
    Process segments in chunks to avoid timeout on large files.
    """
    if len(segments) <= CHUNK_SIZE:
        # Small file: process all at once
        result = await call_qwen_text(prompt, segments, sem)
        return parse_segments(result)

    # Large file: process in chunks
    n_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"    Processing in {n_chunks} chunks of ~{CHUNK_SIZE} segments...", flush=True)

    all_results = []
    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        print(f"    Chunk {chunk_num}/{n_chunks} ({len(chunk)} segments)...", flush=True)

        result = await call_qwen_text(prompt, chunk, sem)
        chunk_results = parse_segments(result)
        all_results.extend(chunk_results)

    return all_results


async def pass_diarization(prompt: str, text: str, sem: asyncio.Semaphore, pass_num: int) -> str:
    """
    Speaker diarization pass (SPEAKER_XX -> actual role).

    Process the file with chunking if needed.
    """
    segments = parse_segments(text)
    print(f"  -> Pass {pass_num}: Diarization ({len(segments)} segments)...", flush=True)

    # Process (with chunking if needed)
    result_segments = await process_chunks(prompt, segments, sem)

    # Verify segment count
    if len(result_segments) != len(segments):
        print(f"  WARNING: Pass {pass_num} segment count mismatch! Input: {len(segments)}, Output: {len(result_segments)}")

    return segments_to_text(result_segments)


async def pass_correction(prompt: str, text: str, sem: asyncio.Semaphore, pass_num: int) -> str:
    """
    Transcription error correction pass.

    Process the file with chunking if needed.
    """
    segments = parse_segments(text)
    print(f"  -> Pass {pass_num}: Correction ({len(segments)} segments)...", flush=True)

    # Process (with chunking if needed)
    result_segments = await process_chunks(prompt, segments, sem)

    # Verify segment count
    if len(result_segments) != len(segments):
        print(f"  WARNING: Pass {pass_num} segment count mismatch! Input: {len(segments)}, Output: {len(result_segments)}")

    return segments_to_text(result_segments)


###############################################################################
# File processing pipeline
###############################################################################
async def process_file(
    path: Path,
    diarization_prompt: str,
    correction_prompt: str,
    req_sem: asyncio.Semaphore,
    failures: list[str],
    benchmark: BenchmarkTimer
):
    out_file = OUTPUT_ROOT / path.parent.name / path.name

    # Count segments for benchmark
    raw_text = path.read_text("utf8")
    segments = parse_segments(raw_text)
    n_segments = len(segments)

    benchmark.start_file(path.name)

    if out_file.exists() and out_file.stat().st_size > 0 and not FORCE:
        print(f"[SKIP] {path.name} already processed")
        benchmark.end_file(path.name, n_segments, "skipped")
        return "skipped"

    print(f"[PROCESSING] {path.name}...", flush=True)
    t0 = time.time()
    try:
        # Pass 1: Diarization
        pass1 = await pass_diarization(diarization_prompt, raw_text, req_sem, pass_num=1)

        # Pass 2: Correction
        pass2 = await pass_correction(correction_prompt, pass1, req_sem, pass_num=2)

        # Pass 3: Diarization refinement
        pass3 = await pass_diarization(diarization_prompt, pass2, req_sem, pass_num=3)

        # Pass 4: Correction refinement
        final = await pass_correction(correction_prompt, pass3, req_sem, pass_num=4)

        # Save result
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(final, "utf8")

        print(f"[OK] {path.name} ({time.time() - t0:.1f}s)")
        benchmark.end_file(path.name, n_segments, "done")
        return "done"
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {path.name} failed: {exc}")
        failures.append(str(path))
        benchmark.end_file(path.name, n_segments, "failed")
        return "failed"


###############################################################################
# Global orchestration
###############################################################################
async def main_async(benchmark: BenchmarkTimer):
    req_sem = asyncio.Semaphore(CONCURRENCY)
    file_sem = asyncio.Semaphore(FILE_CONCURRENCY)
    failures: list[str] = []
    stats = {"done": 0, "skipped": 0, "failed": 0}

    async def enqueue_file(fpath: Path, domain: str):
        await file_sem.acquire()
        try:
            diarization_prompt = DIARIZATION_PROMPTS[domain]
            correction_prompt = CORRECTION_PROMPTS[domain]
            res = await process_file(fpath, diarization_prompt, correction_prompt, req_sem, failures, benchmark)
            stats[res] += 1
        finally:
            file_sem.release()

    tasks = []
    for sub in ("neurochirurgie", "prevention_suicide"):
        for f in (INPUT_ROOT / sub).glob("*.txt"):
            tasks.append(asyncio.create_task(enqueue_file(f, sub)))

    await asyncio.gather(*tasks)
    return stats, failures


###############################################################################
# Entry point
###############################################################################


def main():
    with BenchmarkTimer(
        script_name="qwen_80b_fourpass",
        model_name=MODEL,
        approach="four-pass",
        chunk_size=CHUNK_SIZE,
        concurrency=CONCURRENCY,
        temperature=TEMPERATURE,
        output_dir=PROJECT_ROOT / "results" / "benchmarks",
    ) as benchmark:
        stats, failures = asyncio.run(main_async(benchmark))

        if failures:
            print("\nFailed files:")
            for p in failures:
                print("  -", p)


if __name__ == "__main__":
    main()