import json
import os
import glob
import csv
from jiwer import wer
import re
from collections import defaultdict
from typing import List, Tuple
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from num2words import num2words
import unicodedata
from scipy.optimize import linear_sum_assignment
import numpy as np
from pathlib import Path


# Paths - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MANUAL_TRANSCRIPTIONS = PROJECT_ROOT / "data" / "manual_transcriptions"

# Determine which transcription version to evaluate (baseline, gpt4o, etc.)
# Can be overridden via command line argument
import sys
TRANSCRIPTION_VERSION = sys.argv[1] if len(sys.argv) > 1 else "baseline"

# Set paths based on version
if TRANSCRIPTION_VERSION == "baseline":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "transcriptions" / "whisperx_pyannote"
elif TRANSCRIPTION_VERSION == "qwen_80b":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b"
elif TRANSCRIPTION_VERSION == "qwen_vl_8b":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_vl_8b"
elif TRANSCRIPTION_VERSION == "gpt4omini":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "gpt4omini"
elif TRANSCRIPTION_VERSION == "qwen_80b_twopass_fewshot":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_twopass_fewshot"
elif TRANSCRIPTION_VERSION == "qwen_80b_twopass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_twopass"
elif TRANSCRIPTION_VERSION == "qwen_80b_twopass_reversed":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_twopass_reversed"
elif TRANSCRIPTION_VERSION == "qwen_80b_threepass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_threepass"
elif TRANSCRIPTION_VERSION == "qwen_80b_threepass_reversed":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_threepass_reversed"
elif TRANSCRIPTION_VERSION == "qwen_80b_fourpass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_fourpass"
elif TRANSCRIPTION_VERSION == "qwen_80b_fivepass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_fivepass"
elif TRANSCRIPTION_VERSION == "qwen_80b_sixpass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_sixpass"
elif TRANSCRIPTION_VERSION == "qwen_80b_sevenpass":
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / "qwen_80b_sevenpass"
else:
    # Generic path for custom versions (try post_processed first, then transcriptions)
    GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "post_processed" / TRANSCRIPTION_VERSION
    if not GENERATED_TRANSCRIPTIONS.exists():
        GENERATED_TRANSCRIPTIONS = PROJECT_ROOT / "results" / "transcriptions" / TRANSCRIPTION_VERSION

OUTPUT_COMPARISONS = PROJECT_ROOT / "results" / "evaluations" / "comparisons" / TRANSCRIPTION_VERSION
EVAL_SUMMARY_PATH = PROJECT_ROOT / "results" / "evaluations" / f"eval_summary_{TRANSCRIPTION_VERSION}.json"
EVAL_DETAILED_PATH = PROJECT_ROOT / "results" / "evaluations" / "detailed" / f"eval_detailed_{TRANSCRIPTION_VERSION}.json"

# Create output folders
OUTPUT_COMPARISONS.mkdir(parents=True, exist_ok=True)
EVAL_DETAILED_PATH.parent.mkdir(parents=True, exist_ok=True)

# === JSON Summary ===
eval_results = {
    "neurochirurgie": {
        "global_wer": [],
        "global_der": [],
        "global_wder": [],
        "wer_per_speaker": defaultdict(list),
    },
    "prevention_suicide": {
        "global_wer": [],
        "global_der": [],
        "global_wder": [],
        "wer_per_speaker": defaultdict(list),
    },
}

# === Detailed per-file results for statistical tests ===
detailed_results = {
    "neurochirurgie": {},
    "prevention_suicide": {},
}

# === Outliers (uninterpretable files) ===
outliers = {
    "neurochirurgie": [],
    "prevention_suicide": [],
}


def is_outlier(wer: float, der: float, wder: float) -> Tuple[bool, List[str]]:
    """
    Detect if a file is an outlier (uninterpretable metrics).

    A file is considered an outlier if any metric is:
    - > 100% (1.0): aberrant result
    - < 0% (0.0): physically impossible result

    Returns:
        Tuple (is_outlier, list of reasons)
    """
    reasons = []

    if wer > 1.0:
        reasons.append(f"WER={wer*100:.1f}%>100%")
    if wer < 0:
        reasons.append(f"WER={wer*100:.1f}%<0%")

    if der > 1.0:
        reasons.append(f"DER={der*100:.1f}%>100%")
    if der < 0:
        reasons.append(f"DER={der*100:.1f}%<0%")

    if wder > 1.0:
        reasons.append(f"WDER={wder*100:.1f}%>100%")
    if wder < 0:
        reasons.append(f"WDER={wder*100:.1f}%<0%")

    return len(reasons) > 0, reasons


# Function to load manual transcriptions (.txt files)
def load_manual_transcriptions(neurochirurgie_path, suicide_path):
    manual_transcriptions = {}
    neuro_files = glob.glob(os.path.join(neurochirurgie_path, "*.txt"))
    for file_path in neuro_files:
        base_name = os.path.basename(file_path).replace(".txt", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            manual_transcriptions[base_name] = text_content
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    suicide_files = glob.glob(os.path.join(suicide_path, "*.txt"))
    for file_path in suicide_files:
        base_name = os.path.basename(file_path).replace(".txt", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            manual_transcriptions[base_name] = text_content
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return manual_transcriptions


# Function to load generated transcriptions
def load_generated_transcriptions(folder_path):
    generated_transcriptions = {}
    subfolders = ["neurochirurgie", "prevention_suicide"]
    for subfolder in subfolders:
        full_path = os.path.join(folder_path, subfolder)
        files = glob.glob(os.path.join(full_path, "*.txt"))
        for file_path in files:
            base_name = (
                os.path.basename(file_path)
                .replace(".txt", "")
                .replace("[whisperx_pyannote]", "")
            )
            with open(file_path, "r", encoding="utf-8") as file:
                generated_transcriptions[base_name] = file.read()
    return generated_transcriptions


def convert_numbers_to_words(text):
    """
    Replace any group of digits (e.g., "42") with the word version ("quarante-deux" in French).
    """

    def replacer(match):
        # Convert captured string to integer
        number_as_int = int(match.group(0))
        # Generate the word representation in French
        return num2words(number_as_int, lang="fr")

    # \b\d+\b = all "integer" groups of digits
    pattern = r"\b\d+\b"
    return re.sub(pattern, replacer, text)


# Function to extract text only from transcription
def extract_text_only(text, is_manual=False):
    # Remove timestamps (if still present)
    text = re.sub(
        r"^\s*(?:nan\s*-\s*nan|\d+\.?\d*\s*-\s*\d+\.?\d*)\s*\S*\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = unicodedata.normalize("NFC", text)

    # Remove line breaks
    text = text.replace("\n", " ")

    # Remove asterisks (for manual transcriptions)
    if is_manual:
        text = text.replace("*", "")

    text = convert_numbers_to_words(text)

    # Remove hyphens but concatenate words
    # text = text.replace("-", "")
    text = text.replace("etcaetera", "etc")
    text = re.sub(r"[»«…]", "", text)

    # Remove punctuation except end-of-sentence punctuation
    text = re.sub(r"[,:;]", " ", text)
    # Replace end-of-sentence punctuation with a period
    text = re.sub(r"[.!?]+", ".", text)
    # Replace '
    text = re.sub(r"[’`']", "'", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Trim spaces at the start and end
    text = text.strip()

    # Sentence processing
    sentences = text.split(".")
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        processed_words = []
        for i, word in enumerate(words):
            if is_manual:
                # Replace words containing asterisks
                if "*" in word:
                    processed_words.append("NAME")
                # Replace sequences of 'x' or 'X'
                elif re.fullmatch(r"[xX]{2,}", word, flags=re.UNICODE):
                    processed_words.append("NAME")
                # Replace capitalized words not at the beginning of a sentence
                # elif i != 0 and len(word) > 1 and word[0].isupper():
                #     processed_words.append("NAME")
                else:
                    processed_words.append(word)
            else:
                # Replace proper nouns in the automatic transcription
                if i != 0 and len(word) > 1 and word[0].isupper():
                    processed_words.append("NAME")
                else:
                    processed_words.append(word)
        processed_sentence = " ".join(processed_words)
        processed_sentences.append(processed_sentence)

    # Reassemble text
    text = ". ".join(processed_sentences)

    # Optionally, remove periods
    text = text.replace(".", "")

    # Convert to lowercase
    text = text.lower()

    return text


# Function to preprocess paragraphs by detecting speaker changes
def preprocess_paragraphs_by_speaker(transcription_text, is_manual=False):
    KNOWN_SPEAKERS = {
        "NC",
        "FJ",
        "D",
        "CJ",
        "MA",
        "MS",
        "MX",
        "PS",
        "NP",
        "AUTO",
        "VIG",
        "PAT",
        "INF",
        "AN",
        "PRO",
        "LV",
    }
    paragraphs = []
    current_paragraph = ""
    current_speaker = None
    current_start_time = None
    current_end_time = None
    lines = transcription_text.strip().splitlines()
    for line in lines:
        if is_manual:
            match = re.match(
                r"^\s*(nan|\d+\.?\d*)\s*-\s*(nan|\d+\.?\d*)\s+(\S+)\s*$", line
            )
        else:
            match = re.match(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s+(.+?)\s*$", line)

        if match:
            start_time = match.group(1)
            end_time = match.group(2)
            speaker = match.group(3)
            # Optional: Check if the speaker is in the known list
            if is_manual and speaker not in KNOWN_SPEAKERS:
                speaker = "UNKNOWN"
            if current_paragraph:
                paragraphs.append(
                    {
                        "speaker": current_speaker,
                        "text": current_paragraph.strip(),
                        "start_time": current_start_time,
                        "end_time": current_end_time,
                    }
                )
                current_paragraph = ""
            current_speaker = speaker
            current_start_time = start_time
            current_end_time = end_time
        else:
            current_paragraph += " " + line.strip()
    if current_paragraph:
        paragraphs.append(
            {
                "speaker": current_speaker,
                "text": current_paragraph.strip(),
                "start_time": current_start_time,
                "end_time": current_end_time,
            }
        )
    return paragraphs


# Function to group text by speaker
def group_text_by_speaker(paragraphs):
    grouped_text = defaultdict(str)
    for para in paragraphs:
        speaker = para["speaker"]
        text = para["text"]
        grouped_text[speaker] += " " + text
    return grouped_text


def filter_unknown_speakers(paragraphs):
    """
    Filter paragraphs with speakers marked as "UNKNOWN"
    or with too short duration.

    :param paragraphs: List of paragraphs to analyze
    :param min_duration: Minimum duration in seconds to include a segment
    :return: List of filtered paragraphs
    """
    filtered_paragraphs = []
    for para in paragraphs:
        speaker = para["speaker"]
        # print(f"Processing paragraph: {para}")
        try:
            start_time = float(para["start_time"])
            end_time = float(para["end_time"])
            duration = end_time - start_time
        except ValueError:
            continue  # Skip segments with invalid timestamps

        # Filter unknown speakers and too short segments
        if speaker != "UNKNOWN":
            filtered_paragraphs.append(para)

    return filtered_paragraphs


def extract_nontranscribed_segments(manual_text):
    """
    Extract non-transcribed time segments from the manual transcription.
    """
    nontranscribed_segments = []
    lines = manual_text.strip().split("\n")
    for line in lines:
        match = re.match(r"([\d\.]+)\s*-\s*([\d\.]+)\s*N_T", line.strip())
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            nontranscribed_segments.append((start_time, end_time))
            print(
                f"Match found with : {match}\nAdding segments : {(start_time, end_time)}"
            )
    return nontranscribed_segments


def filter_transcription(transcription, nontranscribed_segments):
    filtered_transcription = []
    lines = transcription.strip().split("\n")
    skip_mode = False  # if True => skip content until new timestamp
    for line in lines:
        match = re.match(r"([\d\.]+)\s*-\s*([\d\.]+)", line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))

            # By default, keep the block
            skip_mode = False

            # Check overlap
            for seg_start, seg_end in nontranscribed_segments:
                if (start_time <= seg_end) and (end_time >= seg_start):
                    skip_mode = True
                    # print(f"Skipping line : {line}")
                    break

            if not skip_mode:
                filtered_transcription.append(line)
        else:
            # Non-timestamped line
            if not skip_mode:
                filtered_transcription.append(line)
            # else:
            #     print(f"Skipping line: {line}")

    return "\n".join(filtered_transcription)


def validate_filtered_transcription(filtered_text, nontranscribed_segments):
    """
    Validate that filtered transcription does not include text from non-transcribed segments.
    """
    for seg_start, seg_end in nontranscribed_segments:
        if re.search(rf"{seg_start}.*{seg_end}", filtered_text):
            raise ValueError(
                f"Filtered transcription still contains non-transcribed segment {seg_start}-{seg_end}"
            )


# === WDER (Word Diarization Error Rate) Calculation ===
# Literature definition: combines transcription errors AND speaker attribution errors

def compute_wder_proper(reference_paragraphs, hypothesis_paragraphs, speaker_mapping, global_wer):
    """
    Compute proper Word Diarization Error Rate (WDER) as defined in literature.

    WDER combines transcription errors (WER) and speaker attribution errors:
    WDER = WER + Speaker_Error_Rate

    Where Speaker_Error_Rate counts correctly transcribed words attributed to wrong speaker.

    This metric is comparable to values reported in papers (typically 2-15% for medical conversations).

    References:
    - Shafey et al. (2019): Joint Speech Recognition and Speaker Diarization
    - Tran et al. (2022): Automatic speech recognition for digital scribes
    """
    from jiwer import wer as compute_wer

    total_words = 0
    speaker_errors = 0

    # Apply automatic -> manual mapping
    hyp_speaker_to_manual = {v: k for k, v in speaker_mapping.items()}

    for hyp_para in hypothesis_paragraphs:
        hyp_text = hyp_para["text"]
        hyp_words = hyp_text.split()
        hyp_speaker = hyp_para["speaker"]
        mapped_speaker = hyp_speaker_to_manual.get(hyp_speaker, "UNKNOWN")

        start_h, end_h = float(hyp_para["start_time"]), float(hyp_para["end_time"])

        # Find matching reference paragraph
        for ref_para in reference_paragraphs:
            start_r, end_r = float(ref_para["start_time"]), float(ref_para["end_time"])
            if abs(start_r - start_h) < 0.5 and abs(end_r - end_h) < 0.5:
                ref_text = ref_para["text"]
                ref_speaker = ref_para["speaker"]

                # Only count speaker errors for correctly transcribed words
                # Approximate: if texts are similar but speaker is wrong
                if mapped_speaker != ref_speaker:
                    # Compute WER for this segment pair
                    try:
                        segment_wer = compute_wer(ref_text, hyp_text)
                        # Words that are correct but have wrong speaker
                        correct_words = len(hyp_words) * (1 - segment_wer)
                        speaker_errors += correct_words
                    except:
                        # If WER computation fails, count all words as speaker errors
                        speaker_errors += len(hyp_words)

                total_words += len(hyp_words)
                break

    if total_words == 0:
        return global_wer

    speaker_error_rate = speaker_errors / total_words

    # WDER = WER + Speaker Error Rate
    # This gives a metric comparable to literature (typically 2-15%)
    wder = global_wer + speaker_error_rate

    return wder


# Main evaluation script
manual_transcriptions = load_manual_transcriptions(
    str(MANUAL_TRANSCRIPTIONS / "neurochirurgie"),
    str(MANUAL_TRANSCRIPTIONS / "prevention_suicide"),
)
generated_transcriptions = load_generated_transcriptions(
    str(GENERATED_TRANSCRIPTIONS)
)

for file_name in generated_transcriptions:
    if file_name in manual_transcriptions:
        manual_transcription = manual_transcriptions[file_name]
        generated_transcription = generated_transcriptions[file_name]

        # Extract non-transcribed segments
        nontranscribed_segments = extract_nontranscribed_segments(manual_transcription)

        # Filter transcriptions
        filtered_manual_transcription = filter_transcription(
            manual_transcription, nontranscribed_segments
        )
        filtered_generated_transcription = filter_transcription(
            generated_transcription, nontranscribed_segments
        )

        # Validate filtered transcriptions
        validate_filtered_transcription(
            filtered_manual_transcription, nontranscribed_segments
        )
        validate_filtered_transcription(
            filtered_generated_transcription, nontranscribed_segments
        )

        # Process transcriptions into paragraphs with speaker information
        manual_paragraphs = preprocess_paragraphs_by_speaker(
            filtered_manual_transcription, is_manual=True
        )
        generated_paragraphs = preprocess_paragraphs_by_speaker(
            filtered_generated_transcription, is_manual=False
        )

        # Apply extract_text_only to each paragraph to clean and normalize text
        for para in manual_paragraphs:
            para["text"] = extract_text_only(para["text"], is_manual=True)
        for para in generated_paragraphs:
            para["text"] = extract_text_only(para["text"], is_manual=False)

        # Group text by speaker
        manual_texts = group_text_by_speaker(manual_paragraphs)
        automatic_texts = group_text_by_speaker(generated_paragraphs)

        # Create CSV file for the comparison
        comparison_file_path = OUTPUT_COMPARISONS / f"{file_name}_comparison.csv"
        with open(
            comparison_file_path, mode="w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.writer(csv_file)

            # Write headers including the new "Automatic Timestamps" column
            writer.writerow(
                [
                    "Manual Transcription",
                    "Manual Speaker",
                    "Manual Timestamps",
                    "Automatic Transcription",
                    "Automatic Speaker",
                    "Automatic Timestamps",
                ]
            )

            # Write rows of transcription, speaker data, and timestamps
            max_length = max(len(manual_paragraphs), len(generated_paragraphs))
            for i in range(max_length):
                if i < len(manual_paragraphs):
                    manual_para = manual_paragraphs[i]["text"]
                    manual_speaker = manual_paragraphs[i]["speaker"]
                    manual_timestamps = f"{manual_paragraphs[i]['start_time']} - {manual_paragraphs[i]['end_time']}"
                else:
                    manual_para = ""
                    manual_speaker = ""
                    manual_timestamps = ""

                if i < len(generated_paragraphs):
                    automatic_para = generated_paragraphs[i]["text"]
                    automatic_speaker = generated_paragraphs[i]["speaker"]
                    automatic_timestamps = f"{generated_paragraphs[i]['start_time']} - {generated_paragraphs[i]['end_time']}"
                else:
                    automatic_para = ""
                    automatic_speaker = ""
                    automatic_timestamps = ""

                writer.writerow(
                    [
                        manual_para,
                        manual_speaker,
                        manual_timestamps,
                        automatic_para,
                        automatic_speaker,
                        automatic_timestamps,
                    ]
                )

        # Calculate global WER and CER using the processed text
        reference_text = " ".join([para["text"] for para in manual_paragraphs])
        hypothesis_text = " ".join([para["text"] for para in generated_paragraphs])
        global_wer = wer(reference_text, hypothesis_text)

        # Find best matching speakers after processing
        # Create WER cost matrix between manual and automatic speakers
        manual_speakers = list(manual_texts.keys())
        automatic_speakers = list(automatic_texts.keys())

        cost_matrix = np.full(
            (len(manual_speakers), len(automatic_speakers)), fill_value=1.0
        )

        for i, manual_speaker in enumerate(manual_speakers):
            for j, automatic_speaker in enumerate(automatic_speakers):
                cost_matrix[i, j] = wer(
                    manual_texts[manual_speaker], automatic_texts[automatic_speaker]
                )

        # Find optimal 1-to-1 assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create mapping
        speaker_mapping = {
            manual_speakers[i]: automatic_speakers[j] for i, j in zip(row_ind, col_ind)
        }

        # Calculate WER for each matched speaker
        wer_per_speaker = {}
        for manual_speaker, automatic_speaker in speaker_mapping.items():
            if automatic_speaker is not None:
                manual_text = manual_texts[manual_speaker]
                automatic_text = automatic_texts[automatic_speaker]
                speaker_wer = wer(manual_text, automatic_text)
                wer_per_speaker[manual_speaker] = (automatic_speaker, speaker_wer)

        # Create reference and hypothesis annotations
        manual_paragraphs = filter_unknown_speakers(manual_paragraphs)
        reference = Annotation()
        for para in manual_paragraphs:
            try:
                start_time = float(para["start_time"])
                end_time = float(para["end_time"])
                if start_time == end_time:
                    continue  # Skip zero-duration segments
            except ValueError:
                continue  # Skip segments with invalid timestamps
            speaker = para["speaker"]
            segment = Segment(start_time, end_time)
            reference[segment] = speaker
        generated_paragraphs = filter_unknown_speakers(generated_paragraphs)
        hypothesis = Annotation()
        for para in generated_paragraphs:
            try:
                start_time = float(para["start_time"])
                end_time = float(para["end_time"])
                if start_time == end_time:
                    continue  # Skip zero-duration segments
            except ValueError:
                continue  # Skip segments with invalid timestamps
            speaker = para["speaker"]
            segment = Segment(start_time, end_time)
            hypothesis[segment] = speaker

        # Compute DER
        metric_der = DiarizationErrorRate()
        der = metric_der(reference, hypothesis)

        # Compute WDER (primary metric)
        wder = compute_wder_proper(manual_paragraphs, generated_paragraphs, speaker_mapping, global_wer)

        # Display results
        print(f"File: {file_name}")
        print(f"Global WER: {global_wer * 100:.2f}%")
        suicide_file_names = set(
            f.stem for f in (GENERATED_TRANSCRIPTIONS / "prevention_suicide").glob("*.txt")
        )

        domain = (
            "prevention_suicide"
            if file_name in suicide_file_names
            else "neurochirurgie"
        )

        for speaker, (matched_speaker, speaker_wer) in wer_per_speaker.items():
            print(
                f"Manual Speaker: {speaker}, Automatic Speaker: {matched_speaker}, WER: {speaker_wer * 100:.2f}%"
            )
            eval_results[domain]["wer_per_speaker"][speaker].append(speaker_wer)
        print(f"DER: {der * 100:.2f}%")
        print(f"WDER: {wder * 100:.2f}%")
        print("=" * 30)

        # Check if file is an outlier
        file_is_outlier, outlier_reasons = is_outlier(global_wer, der, wder)

        if file_is_outlier:
            print(f"  OUTLIER: {', '.join(outlier_reasons)}")
            outliers[domain].append({
                "file": file_name,
                "reasons": outlier_reasons,
                "metrics": {"wer": global_wer, "der": der, "wder": wder}
            })

        eval_results[domain]["global_wer"].append(global_wer)
        eval_results[domain]["global_der"].append(der)
        eval_results[domain]["global_wder"].append(wder)

        # Store detailed per-file results for statistical tests
        detailed_results[domain][file_name] = {
            "wer": global_wer,
            "der": der,
            "wder": wder,
            "is_outlier": file_is_outlier,
            "outlier_reasons": outlier_reasons if file_is_outlier else [],
        }

    else:
        print(f"Manual transcription for file {file_name} not found.")


# Display detected outliers
print("\n" + "=" * 60)
print("DETECTED OUTLIERS (uninterpretable files)")
print("=" * 60)
for domain in ["neurochirurgie", "prevention_suicide"]:
    if outliers[domain]:
        print(f"\n{domain.upper()}:")
        for o in outliers[domain]:
            print(f"  - {o['file']}: {', '.join(o['reasons'])}")
    else:
        print(f"\n{domain.upper()}: No outliers")

# Calculate indices of valid files (non-outliers)
def get_valid_indices(domain: str) -> List[int]:
    """Return indices of non-outlier files."""
    outlier_files = {o["file"] for o in outliers[domain]}
    valid_indices = []
    for i, file_data in enumerate(detailed_results[domain].items()):
        filename, _ = file_data
        if filename not in outlier_files:
            valid_indices.append(i)
    return valid_indices

# Save mean results and standard deviations
summary = {}
for domain in ["neurochirurgie", "prevention_suicide"]:
    wer_vals = eval_results[domain]["global_wer"]
    der_vals = eval_results[domain]["global_der"]
    wder_vals = eval_results[domain]["global_wder"]

    # Filtered values (without outliers)
    outlier_files = {o["file"] for o in outliers[domain]}
    filtered_data = {"wer": [], "der": [], "wder": []}
    for filename, metrics in detailed_results[domain].items():
        if filename not in outlier_files:
            filtered_data["wer"].append(metrics["wer"])
            filtered_data["der"].append(metrics["der"])
            filtered_data["wder"].append(metrics["wder"])

    n_total = len(wer_vals)
    n_valid = len(filtered_data["wer"])
    n_outliers = len(outliers[domain])

    summary[domain] = {
        # Global statistics (all files)
        "all_files": {
            "n_files": n_total,
            "mean_wer": float(np.mean(wer_vals)) if wer_vals else 0.0,
            "std_wer": float(np.std(wer_vals)) if wer_vals else 0.0,
            "mean_der": float(np.mean(der_vals)) if der_vals else 0.0,
            "std_der": float(np.std(der_vals)) if der_vals else 0.0,
            "mean_wder": float(np.mean(wder_vals)) if wder_vals else 0.0,
            "std_wder": float(np.std(wder_vals)) if wder_vals else 0.0,
        },
        # Statistics without outliers (interpretable files)
        "valid_files": {
            "n_files": n_valid,
            "n_outliers_excluded": n_outliers,
            "mean_wer": float(np.mean(filtered_data["wer"])) if filtered_data["wer"] else 0.0,
            "std_wer": float(np.std(filtered_data["wer"])) if filtered_data["wer"] else 0.0,
            "mean_der": float(np.mean(filtered_data["der"])) if filtered_data["der"] else 0.0,
            "std_der": float(np.std(filtered_data["der"])) if filtered_data["der"] else 0.0,
            "mean_wder": float(np.mean(filtered_data["wder"])) if filtered_data["wder"] else 0.0,
            "std_wder": float(np.std(filtered_data["wder"])) if filtered_data["wder"] else 0.0,
        },
        # List of outliers
        "outliers": [
            {"file": o["file"], "reasons": o["reasons"]}
            for o in outliers[domain]
        ],
        # WER per speaker (all files)
        "wer_per_speaker": {
            spk: {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals)) if vals else 0.0,
                "n_files": len(vals)
            }
            for spk, vals in eval_results[domain]["wer_per_speaker"].items()
        },
    }

# Display summary of filtered results
print("\n" + "=" * 60)
print("RESULTS SUMMARY (without outliers)")
print("=" * 60)
for domain in ["neurochirurgie", "prevention_suicide"]:
    valid = summary[domain]["valid_files"]
    n_total = summary[domain]["all_files"]["n_files"]
    n_valid = valid["n_files"]
    print(f"\n{domain.upper()} ({n_valid}/{n_total} files, {valid['n_outliers_excluded']} outliers excluded):")
    print(f"  WER:  {valid['mean_wer']*100:.2f}% ± {valid['std_wer']*100:.2f}%")
    print(f"  DER:  {valid['mean_der']*100:.2f}% ± {valid['std_der']*100:.2f}%")
    print(f"  WDER: {valid['mean_wder']*100:.2f}% ± {valid['std_wder']*100:.2f}%")

with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# Save detailed per-file results for statistical tests
with open(EVAL_DETAILED_PATH, "w", encoding="utf-8") as f:
    json.dump(detailed_results, f, indent=4, ensure_ascii=False)

print(f"\nSummary saved to {EVAL_SUMMARY_PATH}")
print(f"Detailed results saved to {EVAL_DETAILED_PATH}")
