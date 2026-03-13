#!/usr/bin/env python3
"""
LLM-based EMS Radio Dialogue Generator
Generates synthetic EMS transcripts for TTS training.
Supports: OpenAI, Anthropic, Ollama (local)
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Generator
from tqdm import tqdm

# Ensure we can import prompts when run as script
import sys
from pathlib import Path
_phase1_root = Path(__file__).resolve().parent
if str(_phase1_root) not in sys.path:
    sys.path.insert(0, str(_phase1_root))

from prompts.ems_dialogue_prompts import (
    SYSTEM_PROMPT,
    EMS_DIALOGUE_PROMPT,
    EMS_BATCH_PROMPT,
    SCENARIO_PROMPTS,
    CHIEF_COMPLAINTS,
)


def _get_openai_client():
    """Lazy load OpenAI client."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("Install openai: pip install openai")


def _get_anthropic_client():
    """Lazy load Anthropic client."""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")


def generate_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
) -> str:
    """Generate using OpenAI API."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_anthropic(
    prompt: str,
    model: str = "claude-3-haiku-20240307",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
) -> str:
    """Generate using Anthropic API."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.content[0].text.strip()


def generate_ollama(
    prompt: str,
    model: str = "llama3.2",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
) -> str:
    """Generate using local Ollama."""
    try:
        import ollama
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_huggingface(
    prompt: str,
    model: str = "HuggingFaceH4/zephyr-7b-beta",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
    device: str = "auto",
) -> str:
    """Generate using HuggingFace transformers (local)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16 if "cuda" in device else torch.float32, device_map=device
    )
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model_obj.device)
    outputs = model_obj.generate(
        **inputs,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


GENERATORS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "ollama": generate_ollama,
    "huggingface": generate_huggingface,
}


def build_single_prompt(scenario: str, chief_complaint: str) -> str:
    """Build prompt for single utterance generation."""
    scenario_desc = SCENARIO_PROMPTS.get(scenario, "Generate a realistic EMS radio utterance.")
    return EMS_DIALOGUE_PROMPT.format(
        scenario=scenario_desc,
        chief_complaint=chief_complaint or random.choice(CHIEF_COMPLAINTS),
    )


def generate_utterances(
    num_utterances: int = 100,
    provider: str = "ollama",
    model: str = "",
    output_path: str = "text_corpus.jsonl",
    scenarios: Optional[List[str]] = None,
    seed: int = 42,
) -> None:
    """
    Generate EMS utterances and save to JSONL.
    Each line: {"text": "...", "source": "llm", "scenario": "...", "chief_complaint": "..."}
    """
    random.seed(seed)
    scenarios = scenarios or list(SCENARIO_PROMPTS.keys())
    gen_fn = GENERATORS.get(provider)
    if not gen_fn:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(GENERATORS.keys())}")

    model = model or {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3.2",
        "huggingface": "HuggingFaceH4/zephyr-7b-beta",
    }.get(provider, "")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seen = set()

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in tqdm(range(num_utterances), desc="Generating"):
            scenario = random.choice(scenarios)
            chief_complaint = random.choice(CHIEF_COMPLAINTS)
            prompt = build_single_prompt(scenario, chief_complaint)
            try:
                text = gen_fn(prompt, model=model)
                # Deduplicate and filter
                text_clean = " ".join(text.split()).strip()
                if not text_clean or len(text_clean) < 10 or text_clean in seen:
                    continue
                seen.add(text_clean)
                record = {
                    "text": text_clean,
                    "source": "llm",
                    "scenario": scenario,
                    "chief_complaint": chief_complaint,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                tqdm.write(f"Error: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="LLM EMS Dialogue Generator")
    parser.add_argument("--num", type=int, default=100, help="Number of utterances to generate")
    parser.add_argument("--provider", choices=list(GENERATORS.keys()), default="ollama")
    parser.add_argument("--model", type=str, default="", help="Model name (default per provider)")
    parser.add_argument("--output", type=str, default="phase1_output/text_corpus.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_utterances(
        num_utterances=args.num,
        provider=args.provider,
        model=args.model or None,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
