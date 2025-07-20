# Speech2Text

A simple wrapper around DeepSpeech2 for end‑to‑end speech‑to‑text inference.

## Description

This repository demonstrates how to run inference using the DeepSpeech2 model (ASR‑DeepSpeech2) on your own audio files. Follow the steps below to clone, configure, and execute the pre‑trained model.

## Prerequisites

- Python 3.7 or higher  
- `git`  
- `pip`  
- (Optional) Google Colab or any environment with `/content` paths (adjust paths if running locally)

## How to Run

### 1. Clone this repo and DeepSpeech2

```bash
# Clone your Speech2Text project
git clone https://github.com/yegee0/Speech2Text.git
cd Speech2Text
git checkout lm_beam_search
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install gdown
```

### 3. Download pre‑trained models & resources

```bash
# Clean acoustic model
gdown 1NHkGDwOUmSyGqO4zHpbLEOAxokygw6m2

# Alternative acoustic model
gdown 1H_tc1sCFfDFFaElu09DuvXi8aaYnGZif

# Pre‑trained language model
gdown 1-hy2wFlaSU46XhpgEVc0E8lIBcHdOQVq

# Lexicon (pronunciation dictionary)
gdown 1ggUebF7z-k5Fz7mjUWq6NHilV3eHiSb7
```

### 4. Prepare your audio & transcripts

```bash
# Create folders for your test audio and transcripts
mkdir -p /content/my-audio
mkdir -p /content/my-transcripts

# Copy your .wav file(s), e.g. harvard1.wav
cp /content/harvard1.wav /content/my-audio/

# Create an empty transcript file
echo "" > /content/my-transcripts/harvard1.txt
```

> **Note:** If you’re running outside of Colab, replace `/content/...` with your local paths, e.g. `./my-audio` and `./my-transcripts`.

### 5. Run inference

```bash
python inference.py \
  -cn=inference_clean_local \
  '+datasets.test.audio_dir=/content/my-audio' \
  '+datasets.test.transcription_dir=/content/my-transcripts' \
  '++dataloader.batch_size=1'
```

- `-cn` selects the config name (e.g. `inference_clean_local`).  
- Adjust `batch_size` to control parallelism.  
- Transcriptions will be written into your transcripts folder.

## Tips & Troubleshooting

- **Path Errors:** Make sure all directories exist and paths are correct.  
- **Download Issues:** Verify that each `gdown` command completes; you should have four files in the root of `ASR-DeepSpeech2`.  
- **Config Variants:** Check `configs/` for other `-cn` options (e.g. noisy or beam‑search variants).  
- **Batch Size:** Increase for parallel processing, but watch GPU/CPU limits. 
