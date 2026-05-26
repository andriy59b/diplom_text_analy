#!/usr/bin/env python3
"""
================================================================================
LLM OUTPUT TEXT PARAMETERS ANALYSIS (Direct Parameters Version)
Bachelor's Thesis: "Analysis of Output Text Parameters in
Modern Generative Language Models"
Andrii Pentsak, Ivan Franko National University of Lviv, 2026
================================================================================
"""

import os
import sys
import re
import time
import csv
import json
import platform
import subprocess
from typing import List, Dict, Any, Optional

# ── Network patch ─────────────────────────────────────────────────────────────
ollama_host = os.environ.get("OLLAMA_HOST", "")
if "0.0.0.0" in ollama_host:
    port = ollama_host.split(":")[-1] if ":" in ollama_host else "11434"
    os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"

TARGET_MODELS: List[str] = [
    "gemma4:e4b",
    "gemma3:12b",
    "qwen3:14b",
    "granite4.1:8b",
    "phi4:14b",
    "deepseek-r1:1.5b",
    "mistral-nemo:12b",
    "gpt-oss:20b",
]

JSON_FILE = "text_analysis_results.json"
CSV_FILE  = "text_analysis_results.csv"

CONFIGS = [
    {
        "suffix":         "std",
        "label":          "Standard",
        "num_predict":    40000,
        "temperature":    0.9,
        "top_p":          0.95,
        "repeat_penalty": 1.0,
        "system": (
            "You are a scientific writing assistant. "
            "Write maximally long, detailed and comprehensive scientific texts. "
            "Never summarize or shorten your response. "
            "Always elaborate every point in maximum detail. "
            "Continue writing until you have covered every aspect thoroughly."
        ),
    },
    {
        "suffix":         "creative",
        "label":          "Creative",
        "num_predict":    50000,
        "temperature":    1.1,
        "top_p":          0.98,
        "repeat_penalty": 1.0,
        "system": (
            "You are a scientific writing assistant with a creative flair. "
            "Write the longest possible text using diverse vocabulary and "
            "varied sentence structures. Use synonyms extensively — never "
            "repeat the same phrase twice. Expand every concept from multiple "
            "angles with different terminology. "
            "Never stop — keep writing until all tokens are exhausted."
        ),
    },
    {
        "suffix":         "precise",
        "label":          "Precise",
        "num_predict":    35000,
        "temperature":    0.7,
        "top_p":          0.90,
        "repeat_penalty": 1.05,
        "system": (
            "You are a rigorous academic writing assistant. "
            "Write exhaustive, structured scientific content with precise terminology. "
            "Every section must be fully developed with examples and explanations. "
            "Do not summarize. Do not conclude early. "
            "Write every chapter completely before moving to the next."
        ),
    },
]

PROMPTS = {
    "prompt_standard": (
        'Write a comprehensive scientific monograph on the topic '
        '"Large Language Models: Architecture, Training, and Applications".\n\n'
        "Chapters:\n"
        "1. Introduction and history of AI development (at least 5 pages)\n"
        "2. Mathematical foundations of transformer architecture (at least 5 pages)\n"
        "3. LLM training process: pre-training, fine-tuning, RLHF (at least 5 pages)\n"
        "4. Overview of models: GPT, BERT, LLaMA, Mistral, Claude, Gemini (at least 5 pages)\n"
        "5. Text quality evaluation: BLEU, ROUGE, BERTScore, perplexity (at least 5 pages)\n"
        "6. Real-world applications: medicine, education, programming, law (at least 5 pages)\n"
        "7. Problems: hallucinations, bias, safety (at least 5 pages)\n"
        "8. Conclusions and future perspectives (at least 5 pages)\n\n"
        "Write each chapter fully. Do not summarize. Do not skip. "
        "Write as much as possible. No bibliography."
    ),
    "prompt_maxlength": (
        "Use ALL available tokens to write the longest possible scientific monograph "
        'on "Large Language Models: Architecture, Training, and Applications".\n\n'
        "RULES:\n"
        "- Never stop until reaching the absolute token limit\n"
        "- Use synonyms — avoid repeating words\n"
        "- Explain every concept from multiple angles\n"
        "- Diverse sentence structures\n"
        "- Expand every bullet into full paragraphs\n"
        "- No bibliography\n\n"
        "Chapters: history, transformer math, training (RLHF), models (GPT/BERT/LLaMA/"
        "Mistral/Claude/Gemini), evaluation (BLEU/ROUGE/BERTScore), applications, "
        "problems, future.\n\n"
        "Write continuously. Maximize output length."
    ),
    "prompt_detailed": (
        "Write the most detailed academic textbook on Large Language Models. "
        "AT MINIMUM 10 full pages per chapter:\n\n"
        "Ch1 History: from McCulloch-Pitts (1943) to transformers. "
        "Dates, researchers, breakthroughs.\n"
        "Ch2 Math: derive all formulas from first principles. "
        "Linear algebra, probability, attention math.\n"
        "Ch3 Training: BPE/WordPiece tokenization, pre-training, RLHF pipeline, "
        "Constitutional AI.\n"
        "Ch4 Architectures: BERT/GPT/T5. GPT-1 to GPT-4, LLaMA 1/2/3, "
        "Mistral, Claude, Gemini, PaLM.\n"
        "Ch5 Evaluation: BLEU derivation, ROUGE, BERTScore, perplexity, "
        "MMLU, HellaSwag.\n"
        "Ch6 Applications: healthcare, legal, code gen, education.\n"
        "Ch7 Challenges: hallucinations, bias, privacy, energy.\n"
        "Ch8 Future: scaling laws, multimodal, AGI.\n\n"
        "Write everything. No abbreviation. No bibliography. "
        "Continue until all tokens exhausted."
    ),
}

DISCOURSE_MARKERS = [
    "however", "furthermore", "moreover", "therefore", "consequently",
    "in addition", "on the other hand", "for example", "for instance",
    "in conclusion", "thus", "hence", "nevertheless", "nonetheless",
    "in contrast", "similarly", "specifically", "notably", "importantly",
]

# ══════════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def count_words(text):
    return len(text.split())

def count_sentences(text):
    return len([s for s in re.split(r'[.!?]+', text) if s.strip()])

def count_paragraphs(text):
    return len([p for p in text.split('\n\n') if p.strip()])

def type_token_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)

def avg_sentence_length(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 2)

def count_discourse(text):
    t = text.lower()
    return sum(t.count(m) for m in DISCOURSE_MARKERS)

def analyze(text):
    return {
        "char_count":        len(text),
        "word_count":        count_words(text),
        "sentence_count":    count_sentences(text),
        "paragraph_count":   count_paragraphs(text),
        "ttr":               type_token_ratio(text),
        "avg_sentence_len":  avg_sentence_length(text),
        "discourse_markers": count_discourse(text),
    }

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate(base_model: str, prompt: str, cfg: dict) -> dict:
    """Send generation request directly to base model with options."""
    full_prompt = (
        "# Large Language Models: Architecture, Training, and Applications\n\n"
        "## Chapter 1: Introduction and Historical Development\n\n"
        + prompt
    )

    payload = {
        "model":  base_model,
        "prompt": full_prompt,
        "system": cfg["system"],
        "stream": True,
        "raw":    False,
        "options": {
            "num_predict":    cfg["num_predict"],
            "temperature":    cfg["temperature"],
            "top_p":          cfg["top_p"],
            "repeat_penalty": cfg["repeat_penalty"],
            "num_ctx":        131072,
        },
    }

    text        = ""
    token_count = 0
    done_reason = "unknown"
    eval_ns     = 0

    try:
        resp = requests.post(
            OLLAMA_URL, json=payload, stream=True, timeout=28800
        )
        # Check for error early
        if resp.status_code != 200:
            print(f"      [!] Error: HTTP {resp.status_code} - {resp.text}")
            return {"text": "", "tokens_generated": 0, "done_reason": "error", "tokens_per_sec": 0.0}

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    text        += chunk["response"]
                    token_count += 1
                    if token_count % 2000 == 0:
                        print(f"      {token_count:,} tokens / {len(text):,} chars")
                if chunk.get("done"):
                    done_reason = chunk.get("done_reason", "unknown")
                    eval_ns     = chunk.get("eval_duration", 0)
                    print(f"      Stopped: {done_reason} @ {token_count:,} tokens")
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"      Error: {e}")

    tps = token_count / (eval_ns / 1e9) if eval_ns > 0 else 0.0
    return {
        "text":             text,
        "tokens_generated": token_count,
        "done_reason":      done_reason,
        "tokens_per_sec":   round(tps, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> List[dict]:
    try:
        with open(JSON_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Loaded {len(data)} existing results")
        return data
    except FileNotFoundError:
        return []

def save_results(results: List[dict]) -> None:
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fields = [
        "record_id", "base_model", "config_label", "prompt_key",
        "temperature", "top_p", "repeat_penalty", "num_predict",
        "char_count", "word_count", "sentence_count", "paragraph_count",
        "ttr", "avg_sentence_len", "discourse_markers",
        "tokens_generated", "done_reason", "tokens_per_sec",
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> List[str]:
    """Check which TARGET_MODELS are available in Ollama."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        available = [m["name"] for m in resp.json().get("models", [])]
        active = []
        print("[+] Model availability:")
        for m in TARGET_MODELS:
            # Check for exact match or base name match
            found = any(a == m or a == f"{m}:latest" for a in available)
            print(f"    {'✓' if found else '✗'} {m}")
            if found:
                active.append(m)
        return active
    except Exception as e:
        print(f"[-] Cannot reach Ollama: {e}")
        return []

def print_summary(results: List[dict]) -> None:
    if not results:
        return
    print("\n" + "=" * 108)
    print("  RESULTS SUMMARY")
    print("=" * 108)
    print(
        f"{'Base Model':<20} {'Config':<10} {'Prompt':<18} "
        f"{'Chars':>9} {'TTR':>6} {'AvgSen':>7} "
        f"{'Disc':>5} {'TPS':>7} {'Stop':<8}"
    )
    print("-" * 108)
    for r in results:
        print(
            f"{r['base_model']:<20} "
            f"{r['config_label']:<10} "
            f"{r['prompt_key']:<18} "
            f"{r.get('char_count', 0):>9,} "
            f"{r.get('ttr', 0):>6.3f} "
            f"{r.get('avg_sentence_len', 0):>7.1f} "
            f"{r.get('discourse_markers', 0):>5} "
            f"{r.get('tokens_per_sec', 0):>7.1f} "
            f"{r.get('done_reason','?'):<8}"
        )
    print("=" * 108)

def main():
    t0 = time.time()

    print("=" * 65)
    print("  LLM TEXT PARAMETERS ANALYSIS (DIRECT)")
    print(f"  Host: {platform.node()} | {platform.platform()}")
    print("=" * 65)

    # Check GPU
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
        print(f"[+] GPU: {gpu}")
    except Exception:
        print("[+] GPU: not detected via nvidia-smi")

    active_models = check_ollama()
    if not active_models:
        print("[-] No models available. Pull them first with: ollama pull <model>")
        return

    results  = load_results()
    done_ids = {r["record_id"] for r in results}

    for base_model in active_models:
        print(f"\n{'='*65}")
        print(f"  MODEL: {base_model}")
        print(f"{'='*65}")

        for cfg in CONFIGS:
            print(f"\n  Config: {cfg['label']} | T={cfg['temperature']} | max={cfg['num_predict']}")

            for prompt_key, prompt_text in PROMPTS.items():
                record_id = f"{base_model}__{cfg['suffix']}__{prompt_key}"

                if record_id in done_ids:
                    print(f"  [~] Skip {record_id}")
                    continue

                print(f"  ── Prompt: {prompt_key} ──")
                
                gen = generate(base_model, prompt_text, cfg)

                if not gen["text"]:
                    print("  [!] Failed or empty response")
                    continue

                params = analyze(gen["text"])

                record = {
                    "record_id":        record_id,
                    "base_model":       base_model,
                    "config_label":     cfg["label"],
                    "config_suffix":    cfg["suffix"],
                    "prompt_key":       prompt_key,
                    "temperature":      cfg["temperature"],
                    "top_p":            cfg["top_p"],
                    "repeat_penalty":   cfg["repeat_penalty"],
                    "num_predict":      cfg["num_predict"],
                    **params,
                    "tokens_generated": gen["tokens_generated"],
                    "done_reason":      gen["done_reason"],
                    "tokens_per_sec":   gen["tokens_per_sec"],
                    "full_text":        gen["text"],
                }
                results.append(record)
                done_ids.add(record_id)
                
                save_results(results)

                print(f"    Chars: {params['char_count']:,} | TPS: {gen['tokens_per_sec']}")

    print_summary(results)
    elapsed = time.time() - t0
    print(f"\n[+] Total time: {int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}")

if __name__ == "__main__":
    main()