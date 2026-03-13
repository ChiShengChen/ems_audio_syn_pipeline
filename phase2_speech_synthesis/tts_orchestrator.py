#!/usr/bin/env python3
"""
TTS Orchestrator: 70% XTTS, 20% Bark, 10% Edge TTS
Splits corpus and runs each engine, merges manifest.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Optional

# Add pipeline root for imports
import sys
_pipeline_root = Path(__file__).resolve().parent.parent
if str(_pipeline_root) not in sys.path:
    sys.path.insert(0, str(_pipeline_root))

from phase2_speech_synthesis.xtts_voice_cloning import (
    create_speaker_references_from_csv,
    run_xtts_synthesis,
    init_xtts,
    probe_speaker_quality,
    select_top_speakers,
)
from phase2_speech_synthesis.bark_synthesis import run_bark_batch
from phase2_speech_synthesis.edge_tts_synthesis import run_edge_batch


def load_corpus(path: str) -> List[dict]:
    """Load corpus JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def split_corpus(
    records: List[dict],
    xtts_ratio: float = 0.7,
    bark_ratio: float = 0.2,
    edge_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split corpus by ratio. Returns (xtts_records, bark_records, edge_records)."""
    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)
    n = len(shuffled)
    i1 = int(n * xtts_ratio)
    i2 = int(n * (xtts_ratio + bark_ratio))
    return shuffled[:i1], shuffled[i1:i2], shuffled[i2:]


def run_orchestrated(
    corpus_path: str,
    output_base: str = "phase2_output",
    human_csv: Optional[str] = None,
    audio_dirs: Optional[List[str]] = None,
    xtts_ratio: float = 0.7,
    bark_ratio: float = 0.2,
    edge_ratio: float = 0.1,
    max_items: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda",
    skip_bark: bool = False,
    skip_edge: bool = False,
) -> str:
    """
    Run full TTS orchestration.
    Returns path to merged manifest.
    """
    records = load_corpus(corpus_path)
    if max_items:
        records = records[:max_items]
    # Adjust ratios if skipping engines
    _xtts, _bark, _edge = xtts_ratio, bark_ratio, edge_ratio
    if skip_bark:
        _bark = 0
        _xtts = _xtts + bark_ratio  # give to XTTS
    if skip_edge:
        _edge = 0
        _xtts = _xtts + edge_ratio
    xtts_rec, bark_rec, edge_rec = split_corpus(
        records, _xtts, _bark, _edge, seed
    )
    base = Path(output_base)
    base.mkdir(parents=True, exist_ok=True)

    # Write split corpuses
    def write_jsonl(recs: List[dict], p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(xtts_rec, base / "corpus_xtts.jsonl")
    write_jsonl(bark_rec, base / "corpus_bark.jsonl")
    write_jsonl(edge_rec, base / "corpus_edge.jsonl")

    merged_manifest = base / "merged_manifest.jsonl"
    if merged_manifest.exists():
        merged_manifest.unlink()

    # XTTS: extract 20 diverse speakers, probe quality, select top 10
    if xtts_rec and human_csv and audio_dirs:
        ref_dir = str(base / "speaker_references")
        all_refs = create_speaker_references_from_csv(
            human_csv, audio_dirs,
            output_dir=ref_dir,
            max_speakers=20,
        )
        if all_refs:
            tts_model = init_xtts(device)
            good_refs, bad_refs = probe_speaker_quality(
                tts_model, all_refs, device=device)
            top_refs = select_top_speakers(good_refs, top_n=10)
            import json as _json
            with open(str(base / "speaker_references" / "speaker_profiles.json"), "w") as _f:
                _json.dump(top_refs, _f, indent=2)
            print(f"Using {len(top_refs)} quality-filtered speakers for XTTS")
            ref_paths = [r["ref_path"] for r in top_refs]
            run_xtts_synthesis(
                str(base / "corpus_xtts.jsonl"),
                ref_paths,
                output_dir=str(base / "xtts"),
                device=device,
            )
            xtts_manifest = base / "xtts" / "manifest.jsonl"
            if xtts_manifest.exists():
                with open(xtts_manifest, "r") as mf:
                    for line in mf:
                        with open(merged_manifest, "a") as out:
                            out.write(line)
    elif xtts_rec:
        print("Skipping XTTS: need --human_csv and --audio_dirs for speaker refs")

    # Bark
    if bark_rec and not skip_bark:
        run_bark_batch(str(base / "corpus_bark.jsonl"), str(base / "bark"))
        mp = base / "bark" / "manifest.jsonl"
        if mp.exists():
            with open(mp, "r") as mf:
                for line in mf:
                    with open(merged_manifest, "a") as out:
                        out.write(line)

    # Edge
    if edge_rec and not skip_edge:
        run_edge_batch(str(base / "corpus_edge.jsonl"), str(base / "edge"))
        mp = base / "edge" / "manifest.jsonl"
        if mp.exists():
            with open(mp, "r") as mf:
                for line in mf:
                    with open(merged_manifest, "a") as out:
                        out.write(line)

    return str(merged_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="combined_corpus.jsonl")
    parser.add_argument("--output", default="phase2_output")
    parser.add_argument("--human_csv", type=str, help="For XTTS speaker refs")
    parser.add_argument("--audio_dirs", nargs="+", help="Audio dirs for ref extraction")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_bark", action="store_true", help="Skip Bark (if not installed)")
    parser.add_argument("--skip_edge", action="store_true", help="Skip Edge TTS (if network issues)")
    parser.add_argument("--xtts_only", action="store_true", help="Use only XTTS (skip Bark and Edge)")
    args = parser.parse_args()
    skip_bark = args.skip_bark or args.xtts_only
    skip_edge = args.skip_edge or args.xtts_only
    run_orchestrated(
        args.corpus,
        output_base=args.output,
        human_csv=args.human_csv,
        audio_dirs=args.audio_dirs,
        max_items=args.max,
        device=args.device,
        skip_bark=skip_bark,
        skip_edge=skip_edge,
    )
    print(f"Merged manifest: {args.output}/merged_manifest.jsonl")


if __name__ == "__main__":
    main()
