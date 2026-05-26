#!/usr/bin/env python3
"""
================================================================================
LLM OUTPUT TEXT PARAMETERS ANALYSIS SUITE
================================================================================
Description:
    A comprehensive tool for collecting and analyzing output text parameters
    from multiple Large Language Models via the Ollama API and Google Gemini API.

    Based on the benchmarking architecture from:
    t3dr0/ollama_benchmark (AGPLv3)
    "Reusing Obsolete Windows 10 PCs for On-Premises LLM Inference"
    by I. Curington and K. Lano (2026)

    Extended and adapted for the bachelor's thesis:
    "Analysis of Output Text Parameters in Modern Generative Language Models"
    by Andrii Pentsak, Ivan Franko National University of Lviv, 2026

Parameters analyzed per model/config:
    - char_count        : Total character count
    - word_count        : Total word count
    - sentence_count    : Total sentence count
    - paragraph_count   : Total paragraph count
    - ttr               : Type-Token Ratio (lexical diversity)
    - avg_sentence_len  : Average sentence length in words
    - discourse_markers : Count of discourse/cohesion markers
    - done_reason       : Why generation stopped (length / stop)
    - tokens_generated  : Actual tokens generated
    - tokens_per_sec    : Generation speed (tokens/second)

Version: 1.0.0
Author:  Andrii Pentsak (adapted from t3dr0/ollama_benchmark by Ian Curington)
License: AGPLv3
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
import statistics
from typing import List, Dict, Any, Optional, Tuple

# ── Network patch (from original ollama_benchmark) ───────────────────────────
ollama_host: str = os.environ.get("OLLAMA_HOST", "")
if "0.0.0.0" in ollama_host:
    print("\n[!] OLLAMA_HOST targeting '0.0.0.0' — patching to 127.0.0.1")
    port: str = ollama_host.split(":")[-1] if ":" in ollama_host else "11434"
    os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("[-] Missing: pip install requests")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("[-] Missing: pip install ollama")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this section to match your setup
# ══════════════════════════════════════════════════════════════════════════════

# Gemini API key (free from aistudio.google.com)
GEMINI_API_KEY: str = "AIzaSyAFiyL2TDn5uKOx-v9ZIAWbVmuKrTCZlzQ"
GEMINI_URL: str = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# Ollama models to benchmark — change to match what you have pulled
TARGET_MODELS: List[str] = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "mistral:7b",
    "phi4-mini",
]

# Output files
CSV_FILENAME: str  = "text_analysis_results.csv"
JSON_FILENAME: str = "text_analysis_results.json"

# Generation options applied to every Ollama model
GENERATION_OPTIONS: Dict[str, Any] = {
    "num_predict": 96000,
    "temperature": 0.9,
    "top_p":       0.95,
    "repeat_penalty": 1.0,
    "num_ctx":     131072,
}

# EOS stop-tokens per model family
MODEL_STOP_TOKENS: Dict[str, List[str]] = {
    "qwen":    ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<eos>"],
    "llama":   ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
    "mistral": ["</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
    "phi":     ["<|im_end|>", "<|endoftext|>", "<|end|>", "<eos>"],
    "default": ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>"],
}

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS: Dict[str, str] = {
    "prompt_standard": (
        'Write a comprehensive scientific monograph on the topic '
        '"Large Language Models: Architecture, Training, and Applications".\n\n'
        "The monograph must contain the following chapters:\n"
        "1. Introduction and history of AI development (at least 5 pages)\n"
        "2. Mathematical foundations of transformer architecture with formulas (at least 5 pages)\n"
        "3. LLM training process: pre-training, fine-tuning, RLHF (at least 5 pages)\n"
        "4. Overview of modern models: GPT, BERT, LLaMA, Mistral, Claude, Gemini (at least 5 pages)\n"
        "5. Text quality evaluation methods: BLEU, ROUGE, BERTScore, perplexity (at least 5 pages)\n"
        "6. Real-world applications in medicine, education, programming, law (at least 5 pages)\n"
        "7. Problems and limitations: hallucinations, bias, safety (at least 5 pages)\n"
        "8. Conclusions and future perspectives (at least 5 pages)\n\n"
        "Write each chapter completely and in full detail. "
        "Do not summarize. Do not skip any points. "
        "Write as much as possible. "
        "Do NOT write a bibliography or references section."
    ),

    "prompt_maxlength": (
        "You must use ALL available tokens to write the longest possible scientific monograph "
        'on "Large Language Models: Architecture, Training, and Applications".\n\n'
        "CRITICAL RULES:\n"
        "- Use every single token available — do not stop until you reach the absolute token limit\n"
        "- Use synonyms and varied vocabulary — avoid repeating the same words\n"
        "- Explain every concept from multiple angles with different terminology\n"
        "- Use diverse sentence structures to maximize unique word usage\n"
        "- Include extensive examples, analogies, case studies, and elaborations\n"
        "- Never use the same phrase twice — always find alternative expressions\n"
        "- Expand every bullet point into multiple full paragraphs\n"
        "- Do NOT write a bibliography or references section\n\n"
        "Chapters to cover exhaustively:\n"
        "1. Introduction and history of AI development\n"
        "2. Mathematical foundations of transformer architecture\n"
        "3. LLM training: pre-training, fine-tuning, RLHF\n"
        "4. Modern models: GPT series, BERT, LLaMA, Mistral, Claude, Gemini\n"
        "5. Evaluation: BLEU, ROUGE, BERTScore, perplexity, human evaluation\n"
        "6. Applications: medicine, education, programming, law, science\n"
        "7. Problems: hallucinations, bias, safety, energy consumption\n"
        "8. Future: AGI, multimodal models, efficiency improvements\n\n"
        "Write continuously without stopping. Maximize output length."
    ),

    "prompt_detailed": (
        "Act as a world-class professor writing the most detailed academic textbook ever "
        "written on Large Language Models.\n\n"
        'Your textbook "Large Language Models: A Complete Scientific Reference" '
        "must be exhaustive and encyclopedic.\n\n"
        "For EACH chapter write AT MINIMUM 10 full pages of dense academic content:\n\n"
        "Chapter 1 - Historical Foundation: Every milestone from McCulloch-Pitts neuron (1943) "
        "through perceptrons, backpropagation, attention mechanisms to modern transformers.\n\n"
        "Chapter 2 - Mathematical Foundations: Derive every formula from first principles. "
        "Linear algebra, probability, information theory, attention mathematics.\n\n"
        "Chapter 3 - Training Methodology: BPE, WordPiece, SentencePiece tokenization, "
        "pre-training objectives, fine-tuning, RLHF pipeline, Constitutional AI.\n\n"
        "Chapter 4 - Model Architectures: BERT, GPT, T5. Compare GPT-1 through GPT-4, "
        "LLaMA 1/2/3, Mistral, Claude 1/2/3, Gemini, PaLM.\n\n"
        "Chapter 5 - Evaluation: BLEU derivation, ROUGE variants, BERTScore, perplexity, "
        "MMLU, HellaSwag, TruthfulQA.\n\n"
        "Chapter 6 - Applications: Healthcare, legal, code generation, education, science.\n\n"
        "Chapter 7 - Challenges: Hallucinations, bias, adversarial attacks, privacy, "
        "environmental impact.\n\n"
        "Chapter 8 - Future: Scaling laws, emergent abilities, multimodal, AGI.\n\n"
        "Write every word of every chapter. Do not abbreviate. "
        "Continue writing until all tokens are exhausted. "
        "Do NOT write a bibliography or references section."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

DISCOURSE_MARKERS: List[str] = [
    "however", "furthermore", "moreover", "therefore", "consequently",
    "in addition", "on the other hand", "for example", "for instance",
    "in conclusion", "thus", "hence", "nevertheless", "nonetheless",
    "in contrast", "similarly", "specifically", "notably", "importantly",
]


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r'[.!?]+', text) if s.strip()])


def count_paragraphs(text: str) -> int:
    return len([p for p in text.split('\n\n') if p.strip()])


def type_token_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 2)


def count_discourse_markers(text: str) -> int:
    t = text.lower()
    return sum(t.count(m) for m in DISCOURSE_MARKERS)


def analyze_text(text: str) -> Dict[str, Any]:
    """Compute all text parameters for a generated response."""
    return {
        "char_count":        len(text),
        "word_count":        count_words(text),
        "sentence_count":    count_sentences(text),
        "paragraph_count":   count_paragraphs(text),
        "ttr":               type_token_ratio(text),
        "avg_sentence_len":  avg_sentence_length(text),
        "discourse_markers": count_discourse_markers(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM DIAGNOSTICS  (adapted from ollama_benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def get_gpu_info() -> str:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpus = []
        for line in res.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                gpus.append(f"{parts[0].strip()} ({parts[1].strip()} MB VRAM)")
        return ", ".join(gpus)
    except Exception:
        return "Unknown / Non-Nvidia GPU"


def run_environment_checks() -> Tuple[str, str, List[str], Dict[str, str]]:
    print("=" * 65)
    print("  SYSTEM DIAGNOSTICS & INITIALIZATION")
    print("=" * 65)
    hostname = platform.node()
    os_info  = platform.platform()
    gpu_info = get_gpu_info()

    print(f"[+] Hostname:    {hostname}")
    print(f"[+] OS:          {os_info}")
    print(f"[+] GPU:         {gpu_info}")

    try:
        local_manifest = ollama.list()
        model_size_map: Dict[str, str] = {}
        models_list = (
            local_manifest.models
            if hasattr(local_manifest, "models")
            else local_manifest.get("models", [])
        )
        for m in models_list:
            name    = m.model if hasattr(m, "model") else m.get("name", "")
            size_b  = m.size  if hasattr(m, "size")  else m.get("size", 0)
            model_size_map[name] = f"{size_b / 1e9:.1f} GB"
        available = list(model_size_map.keys())
    except Exception as e:
        print(f"\n[-] Cannot reach Ollama API: {e}")
        sys.exit(1)

    print("\n[+] Target model availability:")
    active: List[str] = []
    for model in TARGET_MODELS:
        found = any(a == model or a.startswith(model + ":") for a in available)
        status = "[AVAILABLE]" if found else "[MISSING — run: ollama pull " + model + "]"
        print(f"    {status:50s} {model}")
        if found:
            active.append(model)

    if not active:
        print("[-] No models available. Aborting.")
        sys.exit(1)

    print(f"\n[+] Gemini API configured: {'YES' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'NO (skipping)'}")
    print(f"\n[+] Proceeding with {len(active)} Ollama model(s) + Gemini\n")
    return hostname, gpu_info, active, model_size_map


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_stop_tokens(model_name: str) -> List[str]:
    name = model_name.lower()
    for family, tokens in MODEL_STOP_TOKENS.items():
        if family in name:
            return tokens
    return MODEL_STOP_TOKENS["default"]


def query_ollama(model: str, prompt: str) -> Dict[str, Any]:
    """
    Send a prompt to an Ollama model in streaming mode.
    Returns generated text + performance metadata.
    """
    stop_tokens = get_stop_tokens(model)

    full_prompt = (
        "# Large Language Models: Architecture, Training, and Applications\n\n"
        "## Chapter 1: Introduction and Historical Development\n\n"
        + prompt
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model":  model,
        "prompt": full_prompt,
        "stream": True,
        "raw":    True,
        "options": {**GENERATION_OPTIONS, "stop": stop_tokens},
    }

    full_text    = ""
    token_count  = 0
    done_reason  = "unknown"
    eval_duration_ns = 0

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=14400)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_text   += chunk["response"]
                    token_count += 1
                    if token_count % 1000 == 0:
                        print(f"    ... {token_count:,} tokens / {len(full_text):,} chars")
                if chunk.get("done", False):
                    done_reason      = chunk.get("done_reason", "unknown")
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    print(f"    Stopped: {done_reason} at {token_count:,} tokens")
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"  [!] Error: {e}")

    tps = token_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    return {
        "text":             full_text,
        "tokens_generated": token_count,
        "done_reason":      done_reason,
        "tokens_per_sec":   round(tps, 2),
    }


def query_gemini(prompt: str) -> Dict[str, Any]:
    """Send a prompt to the Gemini 2.0 Flash API."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("  [!] Gemini API key not set — skipping.")
        return {"text": "", "tokens_generated": 0, "done_reason": "skipped", "tokens_per_sec": 0.0}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":    0.9,
            "topP":           0.95,
            "maxOutputTokens": 32768,
        },
    }
    t0 = time.time()
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=300)
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        elapsed = time.time() - t0
        # Gemini doesn't return token count directly — estimate
        est_tokens = len(text.split()) * 4 // 3
        tps = est_tokens / elapsed if elapsed > 0 else 0.0
        print(f"    Generated {len(text):,} chars in {elapsed:.1f}s")
        return {
            "text":             text,
            "tokens_generated": est_tokens,
            "done_reason":      "stop",
            "tokens_per_sec":   round(tps, 2),
        }
    except Exception as e:
        print(f"  [!] Gemini error: {e}")
        return {"text": "", "tokens_generated": 0, "done_reason": "error", "tokens_per_sec": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_results() -> List[Dict[str, Any]]:
    try:
        with open(JSON_FILENAME, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Loaded {len(data)} existing results from {JSON_FILENAME}")
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[!] Could not load existing results: {e}")
        return []


def save_results(results: List[Dict[str, Any]]) -> None:
    # Save full data as JSON (includes full_text)
    with open(JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save summary as CSV (no full_text — cleaner for spreadsheets)
    csv_fields = [
        "config", "model", "source", "prompt",
        "temperature", "top_p", "repeat_penalty",
        "char_count", "word_count", "sentence_count", "paragraph_count",
        "ttr", "avg_sentence_len", "discourse_markers",
        "tokens_generated", "done_reason", "tokens_per_sec",
    ]
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Saved {len(results)} results → {JSON_FILENAME} + {CSV_FILENAME}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 105)
    print("  FINAL SUMMARY — OUTPUT TEXT PARAMETERS")
    print("=" * 105)
    header = (
        f"{'Config':<38} | {'Chars':>9} | {'Words':>7} | "
        f"{'TTR':>6} | {'AvgSent':>7} | {'Disc':>5} | "
        f"{'TPS':>7} | Stop"
    )
    print(header)
    print("-" * 105)
    for r in results:
        if not r.get("char_count"):
            continue
        print(
            f"{r['config']:<38} | "
            f"{r['char_count']:>9,} | "
            f"{r['word_count']:>7,} | "
            f"{r['ttr']:>6.3f} | "
            f"{r['avg_sentence_len']:>7.1f} | "
            f"{r['discourse_markers']:>5} | "
            f"{r['tokens_per_sec']:>7.1f} | "
            f"{r.get('done_reason','?')}"
        )
    print("=" * 105)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    bench_start = time.time()

    hostname, gpu_info, active_models, model_size_map = run_environment_checks()

    results = load_existing_results()
    done_configs = {r["config"] for r in results}

    # ── Ollama models × 3 prompts ────────────────────────────────────────────
    for model in active_models:
        for prompt_key, prompt_text in PROMPTS.items():
            config_name = f"{model}_{prompt_key}"

            if config_name in done_configs:
                print(f"\n[~] Skipping {config_name} — already in results")
                continue

            print(f"\n{'='*65}")
            print(f"  MODEL:  {model}")
            print(f"  PROMPT: {prompt_key}")
            print(f"  CONFIG: {config_name}")
            print(f"{'='*65}")

            gen = query_ollama(model, prompt_text)

            if not gen["text"]:
                print("  [!] Empty response — skipping.")
                continue

            text_params = analyze_text(gen["text"])

            record: Dict[str, Any] = {
                "config":          config_name,
                "model":           model,
                "source":          "ollama",
                "prompt":          prompt_key,
                "temperature":     GENERATION_OPTIONS["temperature"],
                "top_p":           GENERATION_OPTIONS["top_p"],
                "repeat_penalty":  GENERATION_OPTIONS["repeat_penalty"],
                **text_params,
                "tokens_generated": gen["tokens_generated"],
                "done_reason":      gen["done_reason"],
                "tokens_per_sec":   gen["tokens_per_sec"],
                "full_text":        gen["text"],
            }
            results.append(record)
            done_configs.add(config_name)

            # Print immediate result
            print(f"\n  Characters:      {text_params['char_count']:,}")
            print(f"  Words:           {text_params['word_count']:,}")
            print(f"  Sentences:       {text_params['sentence_count']:,}")
            print(f"  Paragraphs:      {text_params['paragraph_count']:,}")
            print(f"  TTR:             {text_params['ttr']:.4f}")
            print(f"  Avg sent len:    {text_params['avg_sentence_len']:.1f} words")
            print(f"  Discourse marks: {text_params['discourse_markers']}")
            print(f"  Tokens/sec:      {gen['tokens_per_sec']:.1f}")
            print(f"  Stop reason:     {gen['done_reason']}")

            # Save after every model×prompt — safe against crashes
            save_results(results)

    # ── Gemini (one prompt — standard) ──────────────────────────────────────
    gemini_config = "gemini-2.0-flash_prompt_standard"
    if gemini_config not in done_configs:
        print(f"\n{'='*65}")
        print("  MODEL:  Gemini 2.0 Flash (API)")
        print("  PROMPT: prompt_standard")
        print(f"{'='*65}")

        gen = query_gemini(PROMPTS["prompt_standard"])

        if gen["text"]:
            text_params = analyze_text(gen["text"])
            record = {
                "config":          gemini_config,
                "model":           "gemini-2.0-flash",
                "source":          "api",
                "prompt":          "prompt_standard",
                "temperature":     0.9,
                "top_p":           0.95,
                "repeat_penalty":  1.0,
                **text_params,
                "tokens_generated": gen["tokens_generated"],
                "done_reason":      gen["done_reason"],
                "tokens_per_sec":   gen["tokens_per_sec"],
                "full_text":        gen["text"],
            }
            results.append(record)
            save_results(results)

    # ── Final summary ────────────────────────────────────────────────────────
    print_summary(results)

    elapsed = time.time() - bench_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\n[+] Total execution time: {h:02d}:{m:02d}:{s:02d}")
    print(f"[+] Results: {JSON_FILENAME}  |  {CSV_FILENAME}")


if __name__ == "__main__":
    main()
