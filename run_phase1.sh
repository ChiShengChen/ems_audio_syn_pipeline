#!/bin/bash
# Phase 1: Text Corpus Generation
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="${PROJECT_ROOT:-/media/meow/One Touch/ems_call}"
OUTPUT="${PROJECT_ROOT}/ems_audio_syn_pipeline/phase1_output"

mkdir -p "$OUTPUT"

echo "=== Phase 1: Text Generation ==="
# 1. LLM generate (use ollama by default - no API key needed)
python3 phase1_text_generation/llm_ems_dialogue_generator.py \
    --num 500 \
    --provider ollama \
    --output "${OUTPUT}/text_corpus.jsonl"

# 2. Collect with human transcripts
python3 phase1_text_generation/corpus_collector.py \
    --human_csv "${PROJECT_ROOT}/vb_ems_anotation/human_anotation_vb.csv" \
    --llm_jsonl "${OUTPUT}/text_corpus.jsonl" \
    --output "${OUTPUT}/combined_corpus.jsonl"

echo "Phase 1 done. Corpus: ${OUTPUT}/combined_corpus.jsonl"
