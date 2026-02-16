#!/usr/bin/env python3
"""
Qwen3-Next-80B Two-Pass Post-Processing with Few-Shot Learning
────────────────────────────────────────────────────────────────────────
Two-pass contextual correction using Qwen3-Next-80B with few-shot examples.

Pass 1: Speaker diarization (SPEAKER_XX -> actual role)
Pass 2: Transcription error correction + anonymization

Features:
• Two-pass processing for better results
• Few-shot learning: adds examples before each request
• Smart resume: skips already processed files
• Async processing with configurable concurrency
• Domain-specific prompts (neurochirurgie, prevention_suicide)

Usage:
    # Normal processing
    python src/post_processing/postprocess_qwen_80b_twopass_fewshot.py

    # Force reprocessing
    export POSTPROCESS_FORCE=1
    python src/post_processing/postprocess_qwen_80b_twopass_fewshot.py
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
OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_twopass_fewshot"
TRAIN_FILES_PATH: Final[Path] = PROJECT_ROOT / "results" / "fine_tuning" / "train_files.txt"
MANUAL_TRANSCRIPTIONS: Final[Path] = PROJECT_ROOT / "data" / "manual_transcriptions"

# Qwen 80B API configuration
MODEL: Final[str] = "Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
BASE_URL: Final[str] = "http://localhost:8000/v1"
API_KEY: Final[str] = "your-api-key"

TEMPERATURE: Final[float] = float(os.getenv("POSTPROCESS_TEMPERATURE", "0"))
MAX_RETRIES: Final[int] = int(os.getenv("POSTPROCESS_RETRIES", "3"))
RETRY_DELAY: Final[int] = 2
CONCURRENCY: Final[int] = int(os.getenv("POSTPROCESS_CONCURRENCY", "2"))
FILE_CONCURRENCY: Final[int] = int(os.getenv("POSTPROCESS_FILE_CONCURRENCY", "1"))
FORCE: Final[bool] = os.getenv("POSTPROCESS_FORCE", "0") == "1"

# Few-shot configuration
N_EXAMPLES: Final[int] = 1  # Number of few-shot examples to include
MAX_EXAMPLE_SEGMENTS: Final[int] = 20  # Max segments per example (to avoid token limits)

# Chunk size for large files (to avoid timeout)
INITIAL_CHUNK_SIZE: Final[int] = int(os.getenv("POSTPROCESS_CHUNK_SIZE", "500"))
MIN_CHUNK_SIZE: Final[int] = int(os.getenv("POSTPROCESS_MIN_CHUNK_SIZE", "100"))
CHUNK_SIZE_REDUCTION: Final[int] = 100  # Reduce by 100 on each timeout

###############################################################################
# OpenAI-compatible Client for Qwen
###############################################################################
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=900.0  # 15 minutes timeout
)

###############################################################################
# Few-Shot Examples Management
###############################################################################

def load_train_files() -> dict[str, list[str]]:
    """
    Load the list of training files from train_files.txt.

    Returns:
        Dict mapping domain to list of filenames
    """
    if not TRAIN_FILES_PATH.exists():
        print(f"[WARNING] {TRAIN_FILES_PATH} not found. Run prepare_finetuning_dataset.py first.")
        return {"neurochirurgie": [], "prevention_suicide": []}

    train_files = {"neurochirurgie": [], "prevention_suicide": []}
    current_domain = None

    with open(TRAIN_FILES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# "):
                current_domain = line[2:]
            elif line and current_domain:
                # Extract filename from "domain/filename.txt"
                filename = line.split("/")[-1]
                train_files[current_domain].append(filename)

    return train_files


def load_few_shot_examples(domain: str, n_examples: int) -> List[Tuple[str, str]]:
    """
    Load few-shot examples (baseline, reference) for a domain.

    Args:
        domain: Domain name (neurochirurgie or prevention_suicide)
        n_examples: Number of examples to load

    Returns:
        List of (baseline_text, reference_text) tuples
    """
    train_files = load_train_files()
    examples = []

    for filename in train_files.get(domain, [])[:n_examples]:
        baseline_path = INPUT_ROOT / domain / filename
        reference_path = MANUAL_TRANSCRIPTIONS / domain / filename

        if baseline_path.exists() and reference_path.exists():
            baseline_text = baseline_path.read_text("utf-8").strip()
            reference_text = reference_path.read_text("utf-8").strip()
            examples.append((baseline_text, reference_text))

        if len(examples) >= n_examples:
            break

    return examples


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


def count_segments(text: str) -> int:
    """Count number of speaker segments (lines with timestamps)."""
    pattern = re.compile(r'^\d+\.?\d*\s*-\s*\d+\.?\d*\s+', re.MULTILINE)
    return len(pattern.findall(text))


def segments_to_text(segments: List[Tuple[str, str, str]]) -> str:
    """Convert segments back to text format."""
    output_lines = []
    for ts, spk, txt in segments:
        output_lines.append(f"{ts} {spk}")
        if txt:
            output_lines.append(txt)
        output_lines.append("")
    return '\n'.join(output_lines).strip()


def truncate_to_n_segments(text: str, n: int) -> str:
    """
    Truncate text to first N segments.

    Args:
        text: Full transcript text
        n: Max number of segments to keep

    Returns:
        Truncated text with first N segments
    """
    segments = parse_segments(text)
    if len(segments) <= n:
        return text

    # Reconstruct text with only first N segments
    result = []
    for timestamp, speaker, content in segments[:n]:
        result.append(f"{timestamp} {speaker}")
        if content:
            result.append(content)
        result.append("")

    return "\n".join(result).strip()


async def call_qwen_with_fewshot(
    prompt: str,
    segments_input: List[Tuple[str, str, str]],
    few_shot_examples: List[Tuple[str, str]],
    sem: asyncio.Semaphore
) -> List[Tuple[str, str, str]]:
    """
    Call Qwen API with few-shot examples, with tolerance for segment count.

    Args:
        prompt: System prompt
        segments_input: Input segments as (timestamp, speaker, text) tuples
        few_shot_examples: List of (input, output) example pairs
        sem: Semaphore for concurrency control

    Returns:
        Result segments as list of tuples
    """
    n_segments = len(segments_input)
    input_text = segments_to_text(segments_input)

    # Build user message with few-shot examples
    user_message = ""

    # Add few-shot examples (truncated to avoid token limits)
    if few_shot_examples:
        user_message += "Voici quelques exemples de corrections attendues :\n\n"
        for i, (example_input, example_output) in enumerate(few_shot_examples, 1):
            # Truncate examples to MAX_EXAMPLE_SEGMENTS
            truncated_input = truncate_to_n_segments(example_input, MAX_EXAMPLE_SEGMENTS)
            truncated_output = truncate_to_n_segments(example_output, MAX_EXAMPLE_SEGMENTS)

            n_ex_segments = count_segments(truncated_input)
            user_message += f"=== EXEMPLE {i} (premiers {n_ex_segments} segments) ===\n"
            user_message += f"Input :\n{truncated_input}\n\n"
            user_message += f"Output attendu :\n{truncated_output}\n\n"

        user_message += "=== FIN DES EXEMPLES ===\n\n"

    # Add actual input to process
    user_message += f"Maintenant, traite cette transcription ({n_segments} segments) :\n\n"
    user_message += f"CRITIQUE : Tu DOIS retourner EXACTEMENT {n_segments} segments (pas plus, pas moins).\n\n"
    user_message += input_text

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=TEMPERATURE,
                )

                result = response.choices[0].message.content.strip()
                result_segments = parse_segments(result)
                diff = abs(len(result_segments) - n_segments)

                if diff > 0:
                    # Allow tolerance of up to 10% or 2 segments, whichever is larger
                    max_tolerance = max(2, int(n_segments * 0.10))

                    if diff <= max_tolerance:
                        print(f"  Segment count mismatch within tolerance (got {len(result_segments)}, expected {n_segments}, diff={diff})")
                        # Truncate if too many segments
                        if len(result_segments) > n_segments:
                            result_segments = result_segments[:n_segments]
                        return result_segments
                    elif attempt < MAX_RETRIES:
                        print(f"  Segment count mismatch (got {len(result_segments)}, expected {n_segments}, diff={diff}), retrying...")
                        await asyncio.sleep(RETRY_DELAY)
                        continue

                return result_segments

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Qwen API error after {MAX_RETRIES} retries: {exc}") from exc
                print(f"  WARNING: Retry {attempt}/{MAX_RETRIES} after error: {exc}")
                await asyncio.sleep(RETRY_DELAY * attempt)


async def process_chunks_with_size(
    prompt: str,
    segments: List[Tuple[str, str, str]],
    few_shot_examples: List[Tuple[str, str]],
    sem: asyncio.Semaphore,
    chunk_size: int
) -> List[Tuple[str, str, str]]:
    """
    Process segments in chunks with a specific chunk size.
    """
    if len(segments) <= chunk_size:
        # Small file: process all at once
        return await call_qwen_with_fewshot(prompt, segments, few_shot_examples, sem)

    # Large file: process in chunks
    n_chunks = (len(segments) + chunk_size - 1) // chunk_size
    print(f"    Processing in {n_chunks} chunks of ~{chunk_size} segments...", flush=True)

    all_results = []
    for i in range(0, len(segments), chunk_size):
        chunk = segments[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"    Chunk {chunk_num}/{n_chunks} ({len(chunk)} segments)...", flush=True)

        chunk_results = await call_qwen_with_fewshot(prompt, chunk, few_shot_examples, sem)
        all_results.extend(chunk_results)

    return all_results


async def process_chunks(
    prompt: str,
    segments: List[Tuple[str, str, str]],
    few_shot_examples: List[Tuple[str, str]],
    sem: asyncio.Semaphore
) -> List[Tuple[str, str, str]]:
    """
    Process segments in chunks with automatic chunk size reduction on timeout.

    Starts with INITIAL_CHUNK_SIZE and reduces by CHUNK_SIZE_REDUCTION
    on each timeout, down to MIN_CHUNK_SIZE.
    """
    current_chunk_size = INITIAL_CHUNK_SIZE

    while current_chunk_size >= MIN_CHUNK_SIZE:
        try:
            return await process_chunks_with_size(prompt, segments, few_shot_examples, sem, current_chunk_size)
        except Exception as exc:
            error_str = str(exc).lower()
            is_timeout = "timeout" in error_str or "timed out" in error_str

            if is_timeout and current_chunk_size > MIN_CHUNK_SIZE:
                new_chunk_size = max(MIN_CHUNK_SIZE, current_chunk_size - CHUNK_SIZE_REDUCTION)
                print(f"    ⚠️ Timeout with chunk_size={current_chunk_size}, reducing to {new_chunk_size}...", flush=True)
                current_chunk_size = new_chunk_size
            else:
                # Not a timeout or already at minimum chunk size, re-raise
                raise

    # Should not reach here, but just in case
    return await process_chunks_with_size(prompt, segments, few_shot_examples, sem, MIN_CHUNK_SIZE)


async def pass1_diarization(
    prompt: str,
    raw: str,
    few_shot_examples: List[Tuple[str, str]],
    sem: asyncio.Semaphore
) -> str:
    """
    Pass 1: Speaker diarization (SPEAKER_XX -> actual role).

    Process the file with chunking for large files.
    """
    segments = parse_segments(raw)
    print(f"  -> Pass 1: Diarization ({len(segments)} segments, {len(few_shot_examples)} examples)...", flush=True)

    # Process with chunking if needed
    result_segments = await process_chunks(prompt, segments, few_shot_examples, sem)

    # Verify segment count with tolerance
    if len(result_segments) != len(segments):
        print(f"  WARNING: Pass 1 segment count mismatch! Input: {len(segments)}, Output: {len(result_segments)}")

    return segments_to_text(result_segments)


async def pass2_correction(
    prompt: str,
    diarized_text: str,
    few_shot_examples: List[Tuple[str, str]],
    sem: asyncio.Semaphore
) -> str:
    """
    Pass 2: Transcription error correction.

    Process the file with chunking for large files.
    """
    segments = parse_segments(diarized_text)
    print(f"  -> Pass 2: Correction ({len(segments)} segments, {len(few_shot_examples)} examples)...", flush=True)

    # Process with chunking if needed
    result_segments = await process_chunks(prompt, segments, few_shot_examples, sem)

    # Verify segment count with tolerance
    if len(result_segments) != len(segments):
        print(f"  WARNING: Pass 2 segment count mismatch! Input: {len(segments)}, Output: {len(result_segments)}")

    return segments_to_text(result_segments)


###############################################################################
# File processing pipeline
###############################################################################
async def process_file(
    path: Path,
    domain: str,
    diarization_prompt: str,
    correction_prompt: str,
    diarization_examples: List[Tuple[str, str]],
    correction_examples: List[Tuple[str, str]],
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
        # Pass 1: Diarization with few-shot examples
        diarized = await pass1_diarization(
            diarization_prompt, raw_text, diarization_examples, req_sem
        )

        # Pass 2: Correction with few-shot examples
        corrected = await pass2_correction(
            correction_prompt, diarized, correction_examples, req_sem
        )

        # Save result
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(corrected, "utf8")

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

    # Load few-shot examples for each domain
    print("=" * 70)
    print("CHARGEMENT DES EXEMPLES FEW-SHOT")
    print("=" * 70)

    few_shot_examples = {}
    for domain in ["neurochirurgie", "prevention_suicide"]:
        examples = load_few_shot_examples(domain, N_EXAMPLES)
        few_shot_examples[domain] = examples
        print(f"  {domain}: {len(examples)} exemples charges")

    print()

    async def enqueue_file(fpath: Path, domain: str):
        await file_sem.acquire()
        try:
            diarization_prompt = DIARIZATION_PROMPTS[domain]
            correction_prompt = CORRECTION_PROMPTS[domain]
            examples = few_shot_examples[domain]

            res = await process_file(
                fpath, domain,
                diarization_prompt, correction_prompt,
                examples, examples,  # Same examples for both passes
                req_sem, failures, benchmark
            )
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
        script_name="qwen_80b_twopass_fewshot",
        model_name=MODEL,
        approach="two-pass-fewshot",
        chunk_size=INITIAL_CHUNK_SIZE,
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