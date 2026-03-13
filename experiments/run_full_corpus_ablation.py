#!/usr/bin/env python3
"""Full corpus A/B/C/D: 598 utts x 4 aug = 2392 clips/group."""
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_augment_ablation import (
    extract_diverse_speaker_refs,
    run_synthesis_for_group,
    run_augmentation_for_group,
    build_original_pipeline,
    build_enhanced_pipeline,
)

CORPUS = str(PIPELINE_ROOT / "phase1_output" / "combined_corpus.jsonl")
OUT = Path(__file__).resolve().parent / "full_corpus_ablation_results"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse or extract refs
ref_dir = OUT / "speaker_refs"
ref_file = ref_dir / "speaker_profiles.json"
if ref_file.exists():
    refs = json.load(open(ref_file))
    ref_paths = [r["ref_path"] for r in refs]
else:
    refs = extract_diverse_speaker_refs(str(ref_dir), max_speakers=20)
    ref_paths = [r["ref_path"] for r in refs]
    ref_dir.mkdir(parents=True, exist_ok=True)
    json.dump(refs, open(ref_file, "w"), indent=2)
print(f"Refs: {len(ref_paths)}")

# Synth 5 and 20 spk
for n in [5, 20]:
    d = OUT / f"synth_{n}spk"
    run_synthesis_for_group(CORPUS, ref_paths[:n], str(d), device="cuda")

# Augment A,B,C,D
configs = [
    ("A_baseline", 5, build_original_pipeline),
    ("B_more_speakers", 20, build_original_pipeline),
    ("C_enhanced_aug", 5, build_enhanced_pipeline),
    ("D_combined", 20, build_enhanced_pipeline),
]
for name, n, pipe_fn in configs:
    m = OUT / f"synth_{n}spk" / "manifest.jsonl"
    aug = OUT / name / "augmented"
    run_augmentation_for_group(str(m), str(aug), pipe_fn, num_variants=4)
    cnt = sum(1 for _ in open(aug / "augmented_manifest.jsonl"))
    print(f"{name}: {cnt} clips")
print("Done: ", OUT)
