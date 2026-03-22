# EMS Audio Synthesis Pipeline

Synthetic speech generation pipeline for **Emergency Medical Services (EMS) radio call** ASR fine-tuning. Generates domain-specific training data for Whisper and similar speech recognition models.

## Overview

| Phase | Description | Output |
|:------|:-------------|:-------|
| **Phase 1** | LLM-based text generation + human transcript collection | `combined_corpus.jsonl` |
| **Phase 2** | TTS synthesis (XTTS 70% + Bark 20% + Edge TTS 10%) | `merged_manifest.jsonl` |
| **Phase 3** | Radio-channel noise augmentation | `augmented_manifest.jsonl` |

The pipeline produces augmented audio clips suitable for Whisper fine-tuning on EMS vocabulary and acoustic conditions.

---

## Requirements

- Python 3.9+
- CUDA GPU (recommended for XTTS)
- API keys for LLM providers (OpenAI / Anthropic) or local [Ollama](https://ollama.ai)

```bash
pip install -r requirements.txt
```

### Optional

- **Bark**: `pip install git+https://github.com/suno-ai/bark`
- **Ollama** (local LLM): no API key needed
- **Whisper prep**: `transformers`, `datasets` (for `prepare_for_whisper.py` and speaker probe)

---

## Project Structure

```
ems_audio_syn_pipeline/
├── phase1_text_generation/     # LLM dialogue generation
│   ├── llm_ems_dialogue_generator.py
│   ├── corpus_collector.py
│   ├── disfluency_injector.py
│   └── prompts/
│       ├── ems_dialogue_prompts.py
│       └── scenario_templates.json
├── phase2_speech_synthesis/     # TTS (XTTS, Bark, Edge)
│   ├── tts_orchestrator.py
│   ├── xtts_voice_cloning.py
│   ├── bark_synthesis.py
│   └── edge_tts_synthesis.py
├── phase3_noise_augmentation/   # Radio channel simulation
│   ├── noise_augmentation.py
│   └── radio_channel_simulator.py
├── experiments/                 # Ablation studies
│   ├── speaker_augment_ablation.py
│   ├── run_ablation.sh
│   └── run_full_corpus_ablation.py
├── utils/
│   ├── audio_utils.py
│   └── torchaudio_patch.py
├── sample_data/                 # Sample corpus (500 utterances)
│   ├── ems_radio_500.jsonl      # JSONL for pipeline
│   └── ems_radio_500_asr.csv    # CSV for reference
├── prepare_for_whisper.py       # Manifest → HuggingFace Dataset
├── finetune_whisper.py          # LoRA fine-tuning for Whisper
├── evaluate_finetuned_whisper.py
├── run_standalone.sh            # Clone-and-run demo (no external data)
├── run_phase1.sh
├── run_phase2.sh
├── run_phase3.sh
├── run_full_pipeline.sh
├── run_improved_pipeline.sh     # + disfluency, v2 augmentation
├── run_whisper_finetuning_synthetic.sh
├── config.example.sh
└── requirements.txt
```

---

## Quick Start

### Standalone Demo (clone and run)

Run the full pipeline with **only in-repo data** — no API keys or external files:

```bash
git clone <repo-url>
cd ems_audio_syn_pipeline
pip install -r requirements.txt
./run_standalone.sh
```

This uses `sample_data/ems_radio_500.jsonl`, synthesizes with Edge TTS only (XTTS/Bark skipped), augments, prepares dataset, fine-tunes 2 epochs, and evaluates. Requires: Python 3.9+, CUDA, `edge-tts`.

---

### Full Setup (external data)

### 1. Configure Paths

Copy `config.example.sh` to `config.sh`, set `PROJECT_ROOT` and other paths, then source before running:

```bash
cp config.example.sh config.sh
# Edit config.sh: export PROJECT_ROOT="/path/to/ems_call"
source config.sh
./run_full_pipeline.sh
```

Or edit the run scripts directly:

- `run_phase1.sh` — `PROJECT_ROOT`, `human_csv`, output paths
- `run_phase2.sh` — `corpus`, `audio_dirs` (speaker reference sources)
- `run_phase3.sh` — `manifest`, `noise_dir`

### 2. Prepare Input Data

| Input | Format | Description |
|:------|:-------|:------------|
| **Human CSV** | `Filename`, `transcript`, `Call Type`, `Tags` | For corpus collection + XTTS speaker refs |
| **Reference audio** | WAV in `audio_dirs` | Filenames must match CSV `Filename` column |
| **Noise samples** | WAV in `noise_samples/ems/` | Optional; radio noise for augmentation |

### 3. Run Pipeline

```bash
# Full pipeline (Phase 1 → 2 → 3)
./run_full_pipeline.sh

# Or step by step
./run_phase1.sh   # Text corpus
./run_phase2.sh   # Speech synthesis
./run_phase3.sh   # Augmentation
```

### 4. Standalone vs full

| Mode | Data | TTS | Usage |
|:-----|:-----|:----|:------|
| **Standalone** | `sample_data/` only | Edge TTS (Bark skipped) | `./run_standalone.sh` |
| **Full** | human_csv + audio_dirs | XTTS + Bark + Edge | Set `PROJECT_ROOT`, run `run_full_pipeline.sh` |

---

## Phase Details

### Phase 1: Text Generation

```bash
# LLM generate (ollama | openai | anthropic)
python3 phase1_text_generation/llm_ems_dialogue_generator.py \
    --num 500 --provider ollama --model llama3.2 \
    --output phase1_output/text_corpus.jsonl

# Collect with human transcripts
python3 phase1_text_generation/corpus_collector.py \
    --human_csv path/to/human_anotation_vb.csv \
    --llm_jsonl phase1_output/text_corpus.jsonl \
    --output phase1_output/combined_corpus.jsonl

# Optional: disfluency injection (uh, um, repetitions)
python3 phase1_text_generation/disfluency_injector.py \
    --input phase1_output/combined_corpus.jsonl \
    --output phase1_output/combined_corpus_disfluent.jsonl \
    --disfluent_ratio 0.5
```

### Phase 2: TTS Synthesis

```bash
python3 phase2_speech_synthesis/tts_orchestrator.py \
    --corpus phase1_output/combined_corpus.jsonl \
    --output phase2_output \
    --human_csv path/to/annotations.csv \
    --audio_dirs dir1 dir2 \
    --device cuda

# Skip Bark/Edge if not installed
# --skip_bark --skip_edge
# XTTS only: --xtts_only
```

### Speaker Profile (XTTS Voice Cloning)

Phase 2 extracts **speaker profiles** from real EMS audio for XTTS voice cloning. The pipeline:

1. **Extract** — `create_speaker_references_from_csv()`: 5s reference clips from human CSV audio, diverse offsets (3s, 8s, 15s, 25s), RMS energy filtering
2. **Probe** — `probe_speaker_quality()`: synthesize probe sentences, transcribe with Whisper, filter hallucination-prone speakers (WER > 1.5)
3. **Select** — `select_top_speakers()`: pick top 10 by lowest probe WER
4. **Output** — `speaker_profiles.json` in `phase2_output/speaker_references/`

**`speaker_profiles.json` format:**

```json
[
  {
    "speaker_id": "spk_07",
    "ref_path": "phase2_output/speaker_references/ref_07.wav",
    "source_file": "202412021022-748072-14744_call_9.wav",
    "offset_sec": 25.0,
    "rms_energy": 0.1247,
    "probe_wer": 0.1
  }
]
```

**Output locations:**

| Path | Description |
|:-----|:-------------|
| `phase2_output/speaker_references/` | Main pipeline (ref_XX.wav + speaker_profiles.json) |
| `experiments/speaker_refs_5/` | Ablation: 5 speakers |
| `experiments/speaker_refs_20/` | Ablation: 20 speakers |
| `experiments/full_corpus_ablation_results/speaker_refs/` | Full corpus ablation |

### Phase 3: Noise Augmentation

Phase 3 applies **audiomentations** + **radio channel simulation** to each clip. Each source clip produces `num_variants` augmented versions.

#### Audiomentations (audiomentations library)

| Transform | Parameters | Probability | Description |
|:----------|:-----------|:-----------:|:------------|
| **BandPassFilter** | 1850 Hz center, BW 1.65–1.7 | 90% | Radio band shaping |
| **ClippingDistortion** | 0–10% percentile | 30% | Simulate overdrive |
| **AddColorNoise** | SNR 12–30 dB, pink/brown (f_decay -4~-1) | 35% | Colored noise (enhanced) |
| **AddGaussianSNR** | SNR 12–35 dB | 35% | White noise (enhanced) |
| **GainTransition** | -10~+4 dB, 0.2–0.8 s | 35% | Gradual gain change (enhanced) |
| **TimeStretch** | 0.8x–1.4x | 60% | Speed perturbation |
| **PitchShift** | -3~+3 semitones | 40% | Pitch variation |
| **Gain** | -12~+6 dB | 50% | Overall level |
| **TimeMask** | 0–15% band part | 40% | SpecAugment-style masking |
| **AddBackgroundNoise** | SNR 3–15 dB | 85% | EMS noise from `noise_dir` (if provided) |
| **Mp3Compression** | 8–32 kbps | 70% | Codec simulation (if `fast_mp3_augment` installed) |

Enhanced mode (`--no_enhanced` to disable): AddColorNoise + GainTransition + conservative SNR. Original mode uses only AddGaussianSNR.

#### Cross-talk overlay

| Effect | Parameters | Description |
|:-------|:-----------|:-------------|
| **Overlay** | SNR 5–15 dB, p=0.2 (default) | Mix another clip from manifest to simulate cross-talk |

#### Radio channel simulator (radio_channel_simulator.py)

| Effect | Parameters | Description |
|:-------|:-----------|:-------------|
| **PTT click** | 20 ms, p=0.7 | Push-to-talk click at start/end |
| **Bandpass** | 300–3400 Hz | Land mobile radio bandwidth |
| **Codec** | AMR-NB (4.75–12.2 kbps) or GSM | Encode/decode via ffmpeg (p=0.5) |
| **Resample degrade** | 8k→16k | Fallback when codec skipped |
| **Signal dropout** | 50 ms blocks, p=0.02/block | Random zero-out (p=0.3 to apply) |
| **Light reverb** | 30 ms delay, decay 0.3 | Ambulance cabin (p=0.4) |

```bash
python3 phase3_noise_augmentation/noise_augmentation.py \
    --manifest phase2_output/merged_manifest.jsonl \
    --output_dir phase3_output \
    --num_variants 4 \
    --noise_dir noise_samples/ems

# Optional: cross-talk overlay
# --crosstalk_p 0.2
```

---

## Whisper Fine-tuning

### Prepare Dataset

```bash
python3 prepare_for_whisper.py \
    --manifest phase3_output/augmented_manifest.jsonl \
    --output_dir whisper_training_data \
    --train_split 0.8 \
    --val_split 0.1
```

Output: `whisper_training_data/{train,validation,test}/` in HuggingFace Dataset format.

### Run Fine-tuning

```bash
# Via run script (prepare + finetune + evaluate)
./run_whisper_finetuning_synthetic.sh

# Or manually
python3 prepare_for_whisper.py --manifest phase3_output/augmented_manifest.jsonl --output_dir whisper_training_data
python3 finetune_whisper.py --dataset_path whisper_training_data --output_dir whisper_finetuned --fp16 --num_train_epochs 3
```

`finetune_whisper.py` supports: `--spec_augment`, `--use_dora`, `--early_stopping`, `--resume_from_checkpoint`.

---

## Experiments: Speaker + Augmentation Ablation

| Experiment | Speakers | Augmentation |
|:-----------|:--------:|:-------------|
| A (baseline) | 5 | Original |
| B (more speakers) | 20 | Original |
| C (enhanced aug) | 5 | + ColorNoise + GainTransition |
| D (combined) | 20 | + ColorNoise + GainTransition |

```bash
cd experiments

# Small corpus (50 utts)
python speaker_augment_ablation.py --phase extract_refs
python speaker_augment_ablation.py --phase synthesize
python speaker_augment_ablation.py --phase augment
python speaker_augment_ablation.py --phase evaluate
# Or: --phase all

# Full corpus (598 utts)
python run_full_corpus_ablation.py
```

---

## Environment Variables

| Variable | Description |
|:---------|:------------|
| `PROJECT_ROOT` | Base path for data/outputs |
| `OPENAI_API_KEY` | For OpenAI LLM (Phase 1) |
| `ANTHROPIC_API_KEY` | For Anthropic LLM (Phase 1) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

---

## Output Formats

### `combined_corpus.jsonl` (Phase 1)

```json
{"text": "...", "source": "llm"|"human", "scenario": "...", "chief_complaint": "..."}
```

### `manifest.jsonl` (Phase 2 / 3)

```json
{"audio_path": "...", "text": "...", "speaker_id": "...", "engine": "xtts"|"bark"|"edge"}
```

Phase 3 augmented manifest may use `audio` instead of `audio_path` for compatibility with `prepare_for_whisper.py`.

---

## License

See [LICENSE](LICENSE) if provided.
