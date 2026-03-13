#!/usr/bin/env python3
"""
Disfluency Injector for EMS Text Corpus
Adds realistic speech imperfections to synthetic transcripts:
  - Filled pauses: "uh", "um", "ah"
  - Word repetitions: "the the patient"
  - Partial words / truncations: "res- rescue 10"
  - Self-corrections: "left-- right arm"
  - Fast-speech contractions: "gonna", "gotta", "wanna"

Usage:
  python disfluency_injector.py --input corpus.jsonl --output corpus_disfluent.jsonl
"""

import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Optional

FILLED_PAUSES = ["uh", "um", "ah", "uh", "um"]

CONTRACTIONS = {
    "going to": "gonna",
    "got to": "gotta",
    "want to": "wanna",
    "let me": "lemme",
    "give me": "gimme",
    "kind of": "kinda",
    "sort of": "sorta",
    "out of": "outta",
    "don't know": "dunno",
}

EMS_CORRECTIONS = [
    ("left", "left-- right"),
    ("right", "right-- left"),
    ("male", "male-- female"),
    ("female", "female-- male"),
    ("conscious", "conscious-- unconscious"),
]


def inject_filled_pause(words: List[str], p: float = 0.08) -> List[str]:
    """Insert filled pauses at random positions."""
    out = []
    for w in words:
        if random.random() < p and out:
            out.append(random.choice(FILLED_PAUSES))
        out.append(w)
    return out


def inject_word_repetition(words: List[str], p: float = 0.04) -> List[str]:
    """Repeat a word (stutter): 'the the patient'."""
    out = []
    for w in words:
        out.append(w)
        if random.random() < p and len(w) > 2:
            out.append(w)
    return out


def inject_partial_word(words: List[str], p: float = 0.03) -> List[str]:
    """Truncate a word: 'res- rescue'."""
    out = []
    for w in words:
        if random.random() < p and len(w) > 4:
            cut = random.randint(2, len(w) // 2)
            out.append(w[:cut] + "-")
        out.append(w)
    return out


def inject_self_correction(text: str, p: float = 0.03) -> str:
    """Insert a self-correction at a random position."""
    if random.random() > p:
        return text
    corr = random.choice(EMS_CORRECTIONS)
    pattern = re.compile(r"\b" + re.escape(corr[0]) + r"\b", re.I)
    if pattern.search(text):
        text = pattern.sub(corr[1], text, count=1)
    return text


def apply_contractions(text: str, p: float = 0.3) -> str:
    """Replace formal phrases with spoken contractions."""
    for formal, casual in CONTRACTIONS.items():
        if formal in text.lower() and random.random() < p:
            text = re.sub(re.escape(formal), casual, text, flags=re.I, count=1)
    return text


def inject_disfluencies(
    text: str,
    pause_p: float = 0.08,
    repeat_p: float = 0.04,
    partial_p: float = 0.03,
    correction_p: float = 0.03,
    contraction_p: float = 0.3,
) -> str:
    """Apply all disfluency types to a text string."""
    text = apply_contractions(text, contraction_p)
    text = inject_self_correction(text, correction_p)
    words = text.split()
    if not words:
        return text
    words = inject_filled_pause(words, pause_p)
    words = inject_word_repetition(words, repeat_p)
    words = inject_partial_word(words, partial_p)
    return " ".join(words)


def process_corpus(
    input_path: str,
    output_path: str,
    keep_original: bool = True,
    disfluent_ratio: float = 0.5,
    seed: int = 42,
) -> int:
    """Process a JSONL corpus: keep originals + add disfluent variants.

    Args:
        input_path: input JSONL (each line has "text" field)
        output_path: output JSONL
        keep_original: if True, write original lines as well
        disfluent_ratio: fraction of lines to also create disfluent copies of
        seed: random seed
    Returns:
        number of output records
    """
    random.seed(seed)
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            if keep_original:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            if random.random() < disfluent_ratio:
                text = rec.get("text", "")
                if not text:
                    continue
                new_text = inject_disfluencies(text)
                new_rec = {**rec, "text": new_text, "disfluent": True}
                f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} records -> {out}")
    print(f"  (originals: {len(records)}, disfluent variants added: {count - len(records) if keep_original else count})")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Inject disfluencies into EMS text corpus",
    )
    parser.add_argument("--input", required=True, help="Input JSONL corpus")
    parser.add_argument("--output", required=True, help="Output JSONL corpus")
    parser.add_argument(
        "--disfluent_ratio", type=float, default=0.5,
        help="Fraction of lines to create disfluent copies (0-1)",
    )
    parser.add_argument(
        "--no_keep_original", action="store_true",
        help="Don't keep original clean text (only disfluent output)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_corpus(
        input_path=args.input,
        output_path=args.output,
        keep_original=not args.no_keep_original,
        disfluent_ratio=args.disfluent_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
