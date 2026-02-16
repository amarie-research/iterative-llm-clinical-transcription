import os
import torch
import torch.serialization
import whisperx
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Fix for PyTorch 2.6+ pickle restrictions with Pyannote models
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
except ImportError:
    pass  # OmegaConf not installed or older PyTorch version

load_dotenv()

# Enable TF32 for faster computation on Ampere GPUs (if available)
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Paths - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
AUDIO_FOLDER = PROJECT_ROOT / "data" / "audio_mp3"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "transcriptions" / "whisperx_pyannote"

neurochirurgie_path = AUDIO_FOLDER / "neurochirurgie"
suicide_path = AUDIO_FOLDER / "prevention_suicide"
neuro_output_folder = OUTPUT_FOLDER / "neurochirurgie"
suicide_output_folder = OUTPUT_FOLDER / "prevention_suicide"

neuro_output_folder.mkdir(parents=True, exist_ok=True)
suicide_output_folder.mkdir(parents=True, exist_ok=True)

def list_audio_files(path):
    return [f.name for f in Path(path).glob("*.mp3")]

neuro_files = list_audio_files(neurochirurgie_path)
suicide_files = list_audio_files(suicide_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load WhisperX model with float16 for GPU memory efficiency
compute_type = "float16" if device == "cuda" else "float32"
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
print(f"Loaded WhisperX model (compute_type={compute_type})")

# Load align model ONCE
align_model, metadata = whisperx.load_align_model(language_code="fr", device=device)
print("Loaded Alignment model")

# Load diarization model on GPU
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

print(f"Loading Pyannote diarization model on {device}...")
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
print(f"✓ Loaded Pyannote diarization model on {device}")

def extract_features(audio_path, segments):
    y, sr = librosa.load(audio_path, sr=16000)
    rows = []

    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        speaker = segment.get("speaker", "UNKNOWN")

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) < 2:
            continue 

        f0 = librosa.yin(segment_audio, fmin=50, fmax=500, sr=sr)
        f0_median = np.median(f0) if len(f0) > 0 else np.nan
        energy = np.mean(segment_audio ** 2)

        words = segment.get("words", [])
        confidences = [w.get("score", 1.0) for w in words if "score" in w]
        conf_mean = np.mean(confidences) if confidences else np.nan

        rows.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "F0_median": f0_median,
            "confidence": conf_mean,
            "energy": energy
        })

    return pd.DataFrame(rows)


def transcribe_and_save(audio_files, folder_path, output_folder):
    for audio_file in audio_files:
        audio_path = Path(folder_path) / audio_file
        print(f"Transcribing {audio_file} with diarization...")

        # Load and transcribe
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, language="fr")

        # Align words
        result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

        # Diarization
        diarize_segments = diarize_model(audio)
        result_diarized = whisperx.assign_word_speakers(diarize_segments, result_aligned)

        # Save transcription
        output_file_name = audio_file.replace(".mp3", "").replace(".wav", "")
        output_file_path = Path(output_folder) / f"{output_file_name}.txt"

        with open(output_file_path, "w", encoding="utf-8") as f:
            for segment in result_diarized["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "")
                f.write(f"{start:.2f} - {end:.2f} {speaker}\n")
                f.write(f"{text}\n\n")

        print(f"Finished transcribing {audio_file}, saved to {output_file_path}")

        # Save features
        features_df = extract_features(str(audio_path), result_diarized["segments"])
        excel_path = Path(output_folder) / f"{output_file_name}_features.xlsx"
        features_df.to_excel(excel_path, index=False)


if __name__ == "__main__":
    print(f"Transcribing {len(neuro_files)} neurochirurgie files...")
    transcribe_and_save(neuro_files, neurochirurgie_path, neuro_output_folder)

    print(f"\nTranscribing {len(suicide_files)} prevention_suicide files...")
    transcribe_and_save(suicide_files, suicide_path, suicide_output_folder)

    print("\nAll transcriptions completed!")
