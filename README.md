# Iterative LLM-based Improvement for French Clinical Interview Transcription and Speaker Diarization

Code repository for the paper: **"Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization"**

---

## Project Structure

```
.
├── src/
│   ├── transcription/
│   │   └── transcribe_baseline.py          # WhisperX + Pyannote baseline
│   ├── post_processing/
│   │   ├── prompts.py                      # Template prompts (customize for your domain)
│   │   ├── domain_adapted_prompts.py       # Domain-specific prompts (paper examples)
│   │   ├── benchmark_utils.py              # Benchmarking and system info utilities
│   │   ├── postprocess_gpt4omini.py        # GPT-4o-mini single-pass
│   │   ├── postprocess_qwen_vl_8b.py       # Qwen VL 8B single-pass
│   │   ├── postprocess_qwen_80b.py         # Qwen 80B single-pass
│   │   ├── postprocess_qwen_80b_twopass.py # 2-pass Diarization-first
│   │   ├── postprocess_qwen_80b_twopass_fewshot.py    # 2-pass with few-shot
│   │   ├── postprocess_qwen_80b_twopass_reversed.py   # 2-pass Correction-first
│   │   ├── postprocess_qwen_80b_threepass.py          # 3-pass (Proposed method)
│   │   ├── postprocess_qwen_80b_threepass_reversed.py # 3-pass Correction-first
│   │   ├── postprocess_qwen_80b_fourpass.py           # 4-pass
│   │   ├── postprocess_qwen_80b_fivepass.py           # 5-pass
│   │   ├── postprocess_qwen_80b_sixpass.py            # 6-pass
│   │   ├── postprocess_qwen_80b_sevenpass.py          # 7-pass
│   │   ├── postprocess_qwen_80b_eightpass.py          # 8-pass
│   │   └── postprocess_qwen_80b_ninepass.py           # 9-pass
│   └── evaluation/
│       ├── evaluate_results.py             # WER, DER, WDER metrics
│       └── statistical_tests.py            # Wilcoxon signed-rank tests
│
├── data/
│   ├── audio_mp3/                          # Raw audio files
│   └── manual_transcriptions/              # Gold standard reference
│
└── results/
    ├── transcriptions/whisperx_pyannote/   # Baseline outputs
    ├── post_processed/                     # LLM-corrected outputs
    └── evaluations/                        # Metrics and statistical tests
```

---

## Pipeline Overview

### 1. Baseline Transcription
- **ASR**: WhisperX large-v3 (French)
- **Diarization**: Pyannote 3.1
- **Output**: Timestamped segments with speaker labels (SPEAKER_00, SPEAKER_01, etc.)

### 2. Three-Pass Architecture (Proposed Method)
1. **Initial Diarization**: Maps generic labels to clinical roles (Patient, Neurosurgeon, etc.)
2. **Lexical Correction**: Corrects ASR errors using clinical context
3. **Diarization Refinement**: Re-evaluates speaker attributions using corrected transcript

### 3. Evaluation
- **WDER** (Word Diarization Error Rate): Primary metric combining WER + speaker attribution errors
- **Statistical significance**: Wilcoxon signed-rank test

---

## Installation

```bash
pip install -r requirements.txt
```

**Note:** For GPU acceleration (recommended for WhisperX), install PyTorch with CUDA support first:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Step 1: Baseline Transcription
```bash
python src/transcription/transcribe_baseline.py
```

### Step 2: LLM Post-Processing
```bash
# Single-pass
python src/post_processing/postprocess_qwen_80b.py

# Three-pass Diarization-first (Proposed method)
python src/post_processing/postprocess_qwen_80b_threepass.py
```

### Step 3: Evaluation
```bash
# Evaluate baseline
python src/evaluation/evaluate_results.py baseline

# Evaluate three-pass
python src/evaluation/evaluate_results.py qwen_80b_threepass

# Run statistical tests
python src/evaluation/statistical_tests.py --threepass-analysis
```

---

## Adapting to New Domains

This pipeline can be adapted to any medical dialogue domain. Follow these steps:

### 1. Create Domain-Specific Prompts

Edit `src/post_processing/domain_adapted_prompts.py` using the templates from `prompts.py`:

- **`DIARIZATION_PROMPTS`**: Define speaker roles for your domain (e.g., Patient, Oncologist, Nurse)
- **`CORRECTION_PROMPTS`**: List domain-specific transcription errors to correct
- **`PROMPTS`**: Combined prompt for single-pass processing (legacy)

Key placeholders to customize:
- `[medical field / specialty]` - Your medical domain
- `[list of possible speaker roles]` - All possible speakers in conversations
- `[describe the typical structure]` - How many speakers, who is always present
- `[list domain-specific transcription errors]` - Common ASR mistakes in your domain

**Note on Language:** The provided `domain_adapted_prompts.py` contains prompts in **French** (for the original French medical recordings). If your transcripts are in another language, you must:
1. Translate the prompts in `domain_adapted_prompts.py` to your target language
2. Update the `user_message` strings in `postprocess_*.py` files (currently in French: "Voici la transcription à traiter...")

### 2. Organize Your Data

Place your data in the expected folder structure:

```
data/
├── audio_mp3/
│   └── your_domain_name/          # Your audio files (.mp3)
└── manual_transcriptions/
    └── your_domain_name/          # Reference transcriptions (.txt)
```

### 3. Update Domain Configuration

In each script, update the domain list to process your data:

```python
# In transcribe_baseline.py and postprocess_*.py files
for domain in ["your_domain_name"]:  # Replace with your domain(s)
    ...

# In evaluate_results.py - update the domain dictionaries
eval_results = {
    "your_domain_name": {...},
}
```

### 4. Match Domain Keys with Prompts

Ensure domain keys in `domain_adapted_prompts.py` match your folder names:

```python
DIARIZATION_PROMPTS = {
    "your_domain_name": """Your customized prompt..."""
}
CORRECTION_PROMPTS = {
    "your_domain_name": """Your customized prompt..."""
}
```

---

## Changing the LLM Model

The post-processing scripts use an **OpenAI-compatible API format**, which is supported by most LLM inference frameworks. This allows you to use local models, self-hosted servers, or cloud APIs.

### Configuration Parameters

In each `postprocess_*.py` script, modify the configuration section:

```python
# Model configuration
MODEL: Final[str] = "your-model-name"              # Model identifier
BASE_URL: Final[str] = "http://localhost:8000/v1"  # API endpoint
API_KEY: Final[str] = "your-api-key"               # Authentication (or dummy for local)

# Processing parameters
TEMPERATURE: Final[float] = 0                      # 0 for deterministic output
MAX_RETRIES: Final[int] = 3                        # Retry attempts on failure
CONCURRENCY: Final[int] = 2                        # Parallel API requests
```

### Compatible Inference Backends

| Backend | Use Case | BASE_URL Example |
|---------|----------|------------------|
| **vLLM** | High-performance local inference | `http://localhost:8000/v1` |
| **Ollama** | Easy local model deployment | `http://localhost:11434/v1` |
| **llama.cpp server** | CPU/lightweight inference | `http://localhost:8080/v1` |
| **TGI (Text Generation Inference)** | HuggingFace models | `http://localhost:8080/v1` |
| **OpenAI API** | Cloud-hosted (GPT-4, etc.) | `https://api.openai.com/v1` |
| **Together AI / Groq** | Cloud-hosted open models | See provider docs |

### Example: Local Model with vLLM

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --port 8000
```

```python
# In postprocess script
MODEL: Final[str] = "Qwen/Qwen2.5-72B-Instruct-AWQ"
BASE_URL: Final[str] = "http://localhost:8000/v1"
API_KEY: Final[str] = "not-needed"  # Local server, no auth required
```

### Example: Local Model with Ollama

```bash
# Pull and run model
ollama pull llama3:70b
```

```python
MODEL: Final[str] = "llama3:70b"
BASE_URL: Final[str] = "http://localhost:11434/v1"
API_KEY: Final[str] = "ollama"
```

### Environment Variables

Configure parameters via environment variables (no code changes needed):

```bash
export POSTPROCESS_TEMPERATURE=0
export POSTPROCESS_RETRIES=3
export POSTPROCESS_CONCURRENCY=2
export POSTPROCESS_CHUNK_SIZE=500
export POSTPROCESS_FORCE=1  # Force reprocessing of existing files
```

---

## Models

- **GPT-4o-mini**: OpenAI commercial model (single-pass only)
- **Qwen3-VL-8B-Instruct-FP8**: Compact vision-language model (single-pass only)
- **Qwen3-Next-80B-A3B-Instruct-AWQ-4bit**: Large open-weight model (all configurations)

---

## Citation

```bibtex
@unpublished{marie:hal-05512777,
  TITLE = {{Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization}},
  AUTHOR = {Marie, Ambre and Bertin, Thomas and Dardenne, Guillaume and Quellec, Gwenol{\'e}},
  URL = {https://hal.science/hal-05512777},
  NOTE = {preprint submitted to Interspeech},
  YEAR = {2026},
  MONTH = Feb,
  KEYWORDS = {clinical interview transcription ; French medical conversations ; speaker attribution ; LLM post-processing ; clinical interview transcription ; automatic speech recognition clinical interview transcription LLM post-processing speaker attribution French medical conversations ; automatic speech recognition},
  PDF = {https://hal.science/hal-05512777v1/file/template.pdf},
  HAL_ID = {hal-05512777},
  HAL_VERSION = {v1},
}
```
