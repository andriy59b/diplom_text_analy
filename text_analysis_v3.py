#!/usr/bin/env python3
"""
================================================================================
LLM OUTPUT TEXT PARAMETERS ANALYSIS — v3.0
Bachelor's Thesis: "Analysis of Output Text Parameters in
Modern Generative Language Models"
Andrii Pentsak, Ivan Franko National University of Lviv, 2026

Based on benchmark architecture from t3dr0/ollama_benchmark (AGPLv3)

OUTPUT FILES:
  generated_texts/          — one .txt file per model×config×prompt (72 files)
  text_analysis_results.csv — all parameters in one table (no full text)
  text_analysis_results.json — full data including generated text
  summary_per_model.csv     — average parameters grouped by model
  summary_per_config.csv    — average parameters grouped by config
  summary_per_prompt.csv    — average parameters grouped by prompt

PARAMETERS COMPUTED:
  char_count          total characters
  word_count          total words
  sentence_count      total sentences
  paragraph_count     total paragraphs
  unique_words        unique word forms
  ttr                 type-token ratio (lexical diversity)
  herdan_c            Herdan's C (vocab richness, length-independent)
  avg_sentence_len    average sentence length in words
  std_sentence_len    std deviation of sentence lengths
  avg_word_len        average word length in characters
  avg_para_len        average paragraph length in sentences
  discourse_markers   count of cohesion/discourse markers
  discourse_per_100w  discourse markers per 100 words (normalized)
  lexical_density     ratio of content words to all words
  long_sentences_pct  % of sentences longer than 30 words
  short_sentences_pct % of sentences shorter than 5 words
  repeat_word_pct     % of unique words appearing more than 3 times
  tokens_generated    tokens produced by model
  done_reason         why generation stopped (stop / length)
  tokens_per_sec      generation speed
================================================================================
"""

import os
import sys
import re
import csv
import json
import math
import time
import platform
import subprocess
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("[-] pip install requests")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"

TARGET_MODELS = [
    "gemma4:e4b",
    "gemma3:12b",
    "qwen3:14b",
    "granite4.1:8b",
    "phi4:14b",
    "deepseek-r1:1.5b",
    "mistral-nemo:12b",
    "gpt-oss:20b",
    "qwen2.5:3b",
]

# Per-model overrides based on v1 analysis results
MODEL_OVERRIDES = {
    "gemma4:e4b":       {"num_ctx": 65536, "num_predict": 96000},
    "qwen3:14b":        {"num_ctx": 65536, "num_predict": 96000},
    "mistral-nemo:12b": {"num_ctx": 65536, "num_predict": 96000},
    "gpt-oss:20b":      {"skip_prompts": ["prompt_detailed"]},
    "deepseek-r1:1.5b": {"num_ctx": 32768, "num_predict": 35000},
    "qwen2.5:3b":       {"num_ctx": 32768, "num_predict": 40000},
}

# Output paths
OUT_DIR        = Path(".")
TEXTS_DIR      = OUT_DIR / "generated_texts"
MAIN_CSV       = OUT_DIR / "text_analysis_results.csv"
MAIN_JSON      = OUT_DIR / "text_analysis_results.json"
MODEL_CSV      = OUT_DIR / "summary_per_model.csv"
CONFIG_CSV     = OUT_DIR / "summary_per_config.csv"
PROMPT_CSV     = OUT_DIR / "summary_per_prompt.csv"

TEXTS_DIR.mkdir(exist_ok=True)

# ── 3 generation configs ──────────────────────────────────────────────────────
CONFIGS = [
    {
        "suffix": "std", "label": "Standard",
        "num_predict": 40000, "temperature": 0.9,
        "top_p": 0.95, "repeat_penalty": 1.0, "num_ctx": 32768,
        "system": (
            "You are a scientific writing assistant. "
            "Write maximally long, detailed and comprehensive scientific texts. "
            "Never summarize or shorten your response. "
            "Always elaborate every point in maximum detail. "
            "Continue writing until you have covered every aspect thoroughly."
        ),
    },
    {
        "suffix": "creative", "label": "Creative",
        "num_predict": 50000, "temperature": 1.1,
        "top_p": 0.98, "repeat_penalty": 1.0, "num_ctx": 32768,
        "system": (
            "You are a scientific writing assistant with a creative flair. "
            "Write the longest possible text using diverse vocabulary. "
            "Use synonyms extensively — never repeat the same phrase twice. "
            "Expand every concept from multiple angles. "
            "Never stop — keep writing until all tokens are exhausted."
        ),
    },
    {
        "suffix": "precise", "label": "Precise",
        "num_predict": 35000, "temperature": 0.7,
        "top_p": 0.90, "repeat_penalty": 1.05, "num_ctx": 32768,
        "system": (
            "You are a rigorous academic writing assistant. "
            "Write exhaustive, structured scientific content with precise terminology. "
            "Every section must be fully developed with examples and explanations. "
            "Do not summarize. Do not conclude early. "
            "Write every chapter completely before moving to the next."
        ),
    },
]

# ── 3 prompts ─────────────────────────────────────────────────────────────────
PROMPTS = {
    "prompt_standard": (
        "Write a comprehensive scientific monograph on the topic "
        "'Large Language Models: Architecture, Training, and Applications'.\n\n"
        "Chapters:\n"
        "1. Introduction and history of AI development (at least 5 pages)\n"
        "2. Mathematical foundations of transformer architecture (at least 5 pages)\n"
        "3. LLM training process: pre-training, fine-tuning, RLHF (at least 5 pages)\n"
        "4. Overview of models: GPT, BERT, LLaMA, Mistral, Claude, Gemini (at least 5 pages)\n"
        "5. Text quality evaluation: BLEU, ROUGE, BERTScore, perplexity (at least 5 pages)\n"
        "6. Real-world applications: medicine, education, programming, law (at least 5 pages)\n"
        "7. Problems: hallucinations, bias, safety (at least 5 pages)\n"
        "8. Conclusions and future perspectives (at least 5 pages)\n\n"
        "Write each chapter fully. Do not summarize. Write as much as possible. No bibliography."
    ),
    "prompt_maxlength": (
        "Use ALL available tokens to write the longest possible scientific monograph "
        "on 'Large Language Models: Architecture, Training, and Applications'.\n\n"
        "CRITICAL RULES:\n"
        "- Never stop until reaching the absolute token limit\n"
        "- Use synonyms — avoid repeating words\n"
        "- Explain every concept from multiple angles\n"
        "- Expand every bullet into full paragraphs\n"
        "- No bibliography\n\n"
        "Chapters: history, transformer math, training (RLHF), "
        "models (GPT/BERT/LLaMA/Mistral/Claude/Gemini), "
        "evaluation (BLEU/ROUGE/BERTScore), applications, problems, future.\n\n"
        "Write continuously. Maximize output length."
    ),
    "prompt_detailed": (
        "Write the most detailed academic textbook on Large Language Models. "
        "AT MINIMUM 10 full pages per chapter:\n\n"
        "Ch1 History: from McCulloch-Pitts (1943) to modern transformers.\n"
        "Ch2 Math: derive all formulas. Linear algebra, probability, attention.\n"
        "Ch3 Training: BPE tokenization, pre-training, RLHF pipeline.\n"
        "Ch4 Architectures: BERT/GPT/T5. GPT-1 to GPT-4, LLaMA, Mistral, Claude, Gemini.\n"
        "Ch5 Evaluation: BLEU, ROUGE, BERTScore, perplexity, MMLU.\n"
        "Ch6 Applications: healthcare, legal, code generation, education.\n"
        "Ch7 Challenges: hallucinations, bias, privacy, energy impact.\n"
        "Ch8 Future: scaling laws, multimodal, AGI.\n\n"
        "Write everything. No bibliography. Continue until all tokens exhausted."
    ),
}

# ── Discourse markers ─────────────────────────────────────────────────────────
DISCOURSE_MARKERS = [
    "however", "furthermore", "moreover", "therefore", "consequently",
    "in addition", "on the other hand", "for example", "for instance",
    "in conclusion", "thus", "hence", "nevertheless", "nonetheless",
    "in contrast", "similarly", "specifically", "notably", "importantly",
    "as a result", "in summary", "to summarize", "in particular",
    "additionally", "subsequently", "finally", "firstly", "secondly",
    "in fact", "indeed", "despite", "although", "whereas",
]

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "this","that","these","those","it","its","as","if","not","also","than",
    "then","so","such","more","most","some","any","all","which","who","what",
    "when","where","how","their","they","them","we","our","you","your","he",
    "she","his","her","i","my","me","can","shall","very",
}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def get_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def get_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]

def get_paragraphs(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def analyze(text: str) -> Dict[str, Any]:
    words      = get_words(text)
    sentences  = get_sentences(text)
    paragraphs = get_paragraphs(text)
    t_lower    = text.lower()

    n_words = len(words)
    n_sents = len(sentences)
    unique  = len(set(words))

    sent_lengths  = [len(get_words(s)) for s in sentences]
    word_lengths  = [len(w) for w in words]
    para_sents    = [len(get_sentences(p)) for p in paragraphs]

    # TTR
    ttr = round(unique / n_words, 4) if n_words > 0 else 0.0

    # Herdan's C
    herdan_c = round(
        math.log(unique) / math.log(n_words), 4
    ) if n_words > 1 and unique > 1 else 0.0

    # Sentence stats
    avg_sent = round(statistics.mean(sent_lengths), 2) if sent_lengths else 0.0
    std_sent = round(statistics.stdev(sent_lengths), 2) if len(sent_lengths) > 1 else 0.0

    # Word length
    avg_word = round(statistics.mean(word_lengths), 2) if word_lengths else 0.0

    # Paragraph length
    avg_para = round(statistics.mean(para_sents), 2) if para_sents else 0.0

    # Discourse markers
    disc_count  = sum(t_lower.count(m) for m in DISCOURSE_MARKERS)
    disc_per100 = round(disc_count / n_words * 100, 2) if n_words > 0 else 0.0

    # Lexical density
    content_words   = [w for w in words if w not in STOPWORDS]
    lex_density     = round(len(content_words) / n_words, 4) if n_words > 0 else 0.0

    # Long / short sentences
    long_pct  = round(sum(1 for l in sent_lengths if l > 30) / n_sents * 100, 2) if n_sents > 0 else 0.0
    short_pct = round(sum(1 for l in sent_lengths if l < 5)  / n_sents * 100, 2) if n_sents > 0 else 0.0

    # Repeat words
    freq = Counter(words)
    repeat_pct = round(
        sum(1 for w, c in freq.items() if c > 3) / unique * 100, 2
    ) if unique > 0 else 0.0

    return {
        "char_count":          len(text),
        "word_count":          n_words,
        "sentence_count":      n_sents,
        "paragraph_count":     len(paragraphs),
        "unique_words":        unique,
        "ttr":                 ttr,
        "herdan_c":            herdan_c,
        "avg_sentence_len":    avg_sent,
        "std_sentence_len":    std_sent,
        "avg_word_len":        avg_word,
        "avg_para_len":        avg_para,
        "discourse_markers":   disc_count,
        "discourse_per_100w":  disc_per100,
        "lexical_density":     lex_density,
        "long_sentences_pct":  long_pct,
        "short_sentences_pct": short_pct,
        "repeat_word_pct":     repeat_pct,
    }

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate(model: str, prompt: str, cfg: Dict) -> Dict:
    ov           = MODEL_OVERRIDES.get(model, {})
    num_predict  = ov.get("num_predict", cfg["num_predict"])
    num_ctx      = ov.get("num_ctx",     cfg["num_ctx"])

    full_prompt = (
        "# Large Language Models: Architecture, Training, and Applications\n\n"
        "## Chapter 1: Introduction and Historical Development\n\n"
        + prompt
    )

    payload = {
        "model":  model,
        "prompt": full_prompt,
        "system": cfg["system"],
        "stream": True,
        "options": {
            "num_predict":    num_predict,
            "temperature":    cfg["temperature"],
            "top_p":          cfg["top_p"],
            "repeat_penalty": cfg["repeat_penalty"],
            "num_ctx":        num_ctx,
        },
    }

    text = ""
    tokens = 0
    done_reason = "unknown"
    eval_ns = 0

    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=28800)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    text   += chunk["response"]
                    tokens += 1
                    if tokens % 2000 == 0:
                        print(f"      {tokens:,} tokens / {len(text):,} chars")
                if chunk.get("done"):
                    done_reason = chunk.get("done_reason", "unknown")
                    eval_ns     = chunk.get("eval_duration", 0)
                    print(f"      Stopped: {done_reason} @ {tokens:,} tokens / {len(text):,} chars")
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"      Error: {e}")

    tps = tokens / (eval_ns / 1e9) if eval_ns > 0 else 0.0
    return {
        "text":             text,
        "tokens_generated": tokens,
        "done_reason":      done_reason,
        "tokens_per_sec":   round(tps, 2),
        "num_predict_used": num_predict,
        "num_ctx_used":     num_ctx,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

MAIN_FIELDS = [
    "record_id", "model", "config", "config_label", "prompt",
    "temperature", "top_p", "repeat_penalty", "num_predict_used", "num_ctx_used",
    "tokens_generated", "done_reason", "tokens_per_sec",
    "char_count", "word_count", "sentence_count", "paragraph_count",
    "unique_words", "ttr", "herdan_c",
    "avg_sentence_len", "std_sentence_len", "avg_word_len", "avg_para_len",
    "discourse_markers", "discourse_per_100w",
    "lexical_density", "long_sentences_pct", "short_sentences_pct",
    "repeat_word_pct", "txt_file",
]

def load_results() -> List[Dict]:
    if MAIN_JSON.exists():
        try:
            with open(MAIN_JSON, encoding="utf-8") as f:
                data = json.load(f)
            print(f"[+] Loaded {len(data)} existing results")
            return data
        except Exception:
            pass
    return []

def save_all(records: List[Dict]) -> None:
    """Save JSON, main CSV, and three summary CSVs."""

    # ── JSON (without full_text to save space) ────────────────────────────────
    slim = [{k: v for k, v in r.items() if k != "full_text"} for r in records]
    with open(MAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    # ── Main CSV ──────────────────────────────────────────────────────────────
    with open(MAIN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MAIN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # ── Summary helpers ───────────────────────────────────────────────────────
    num_params = [
        "char_count", "word_count", "unique_words", "ttr", "herdan_c",
        "avg_sentence_len", "std_sentence_len", "avg_word_len",
        "discourse_markers", "discourse_per_100w",
        "lexical_density", "long_sentences_pct", "short_sentences_pct",
        "repeat_word_pct", "tokens_per_sec",
    ]

    def group_summary(records, group_key, out_path):
        groups = defaultdict(list)
        for r in records:
            groups[r.get(group_key, "?")].append(r)
        rows = []
        for key, recs in sorted(groups.items()):
            row = {group_key: key, "count": len(recs)}
            for p in num_params:
                vals = [r[p] for r in recs if isinstance(r.get(p), (int, float))]
                row[f"{p}_mean"] = round(statistics.mean(vals), 4) if vals else ""
                row[f"{p}_std"]  = round(statistics.stdev(vals), 4) if len(vals) > 1 else ""
            rows.append(row)
        fields = [group_key, "count"] + [f"{p}{s}" for p in num_params for s in ("_mean", "_std")]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    group_summary(records, "model",        MODEL_CSV)
    group_summary(records, "config_label", CONFIG_CSV)
    group_summary(records, "prompt",       PROMPT_CSV)


def save_txt(record_id: str, text: str) -> str:
    safe = record_id.replace(":", "-").replace("/", "-").replace("\\", "-")
    path = TEXTS_DIR / f"{safe}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def check_ollama() -> List[str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        available = [m["name"] for m in resp.json().get("models", [])]
        active = []
        print("[+] Models:")
        for m in TARGET_MODELS:
            found = any(a == m or a.startswith(m.split(":")[0]) for a in available)
            print(f"    {'✓' if found else '✗ MISSING'} {m}")
            if found:
                active.append(m)
        return active
    except Exception as e:
        print(f"[-] Ollama error: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(records: List[Dict]) -> None:
    valid = [r for r in records if r.get("char_count", 0) > 0]
    print("\n" + "=" * 122)
    print("  RESULTS SUMMARY")
    print("=" * 122)
    print(
        f"{'Model':<22} {'Config':<10} {'Prompt':<18} "
        f"{'Chars':>9} {'Words':>7} {'TTR':>6} {'HerdC':>6} "
        f"{'AvgSen':>7} {'Disc/100':>9} {'LexDen':>7} {'TPS':>7}  Stop"
    )
    print("-" * 122)
    for r in valid:
        print(
            f"{r['model']:<22} {r['config_label']:<10} {r['prompt']:<18} "
            f"{r['char_count']:>9,} {r['word_count']:>7,} "
            f"{r['ttr']:>6.3f} {r['herdan_c']:>6.3f} "
            f"{r['avg_sentence_len']:>7.1f} {r['discourse_per_100w']:>9.2f} "
            f"{r['lexical_density']:>7.3f} {r['tokens_per_sec']:>7.1f}  "
            f"{r.get('done_reason', '?')}"
        )
    print("=" * 122)
    total_chars = sum(r.get("char_count", 0) for r in valid)
    stop_n   = sum(1 for r in valid if r.get("done_reason") == "stop")
    length_n = sum(1 for r in valid if r.get("done_reason") == "length")
    print(f"\nTotal records : {len(valid)}")
    print(f"Total chars   : {total_chars:,}")
    print(f"Stop (natural): {stop_n}   Length (limit): {length_n}")

    print(f"\n{'='*60}")
    print("  OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  {TEXTS_DIR}/          — {len(list(TEXTS_DIR.glob('*.txt')))} txt files")
    print(f"  {MAIN_CSV}            — main parameter table")
    print(f"  {MAIN_JSON}           — full data (JSON)")
    print(f"  {MODEL_CSV}           — averages per model")
    print(f"  {CONFIG_CSV}          — averages per config")
    print(f"  {PROMPT_CSV}          — averages per prompt")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 65)
    print("  LLM TEXT PARAMETERS ANALYSIS v3.0")
    print(f"  Host: {platform.node()}")
    print("=" * 65)

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"], encoding="utf-8").strip()
        print(f"[+] GPU: {gpu}")
    except Exception:
        pass

    active_models = check_ollama()
    if not active_models:
        print("[-] No models. Run: ollama pull <model>")
        return

    total = sum(
        len(CONFIGS) * (len(PROMPTS) - len(MODEL_OVERRIDES.get(m, {}).get("skip_prompts", [])))
        for m in active_models
    )
    print(f"\n[+] {len(active_models)} models | ~{total} records total")

    records  = load_results()
    done_ids = {r["record_id"] for r in records}

    for model in active_models:
        ov           = MODEL_OVERRIDES.get(model, {})
        skip_prompts = ov.get("skip_prompts", [])

        print(f"\n{'='*65}")
        print(f"  MODEL: {model}")
        if skip_prompts:
            print(f"  Skipping: {skip_prompts}")
        print(f"{'='*65}")

        for cfg in CONFIGS:
            print(f"\n  [{cfg['label']}] T={cfg['temperature']} "
                  f"top_p={cfg['top_p']} "
                  f"max_tok={ov.get('num_predict', cfg['num_predict'])} "
                  f"ctx={ov.get('num_ctx', cfg['num_ctx'])}")

            for prompt_key, prompt_text in PROMPTS.items():
                if prompt_key in skip_prompts:
                    print(f"    [x] Skip {prompt_key}")
                    continue

                record_id = f"{model}__{cfg['suffix']}__{prompt_key}"

                # Skip if txt already exists and result recorded
                txt_path = TEXTS_DIR / f"{record_id.replace(':', '-').replace('/', '-')}.txt"
                if record_id in done_ids and txt_path.exists():
                    print(f"    [~] Skip {record_id}")
                    continue

                print(f"\n    Prompt: {prompt_key}")

                gen = generate(model, prompt_text, cfg)

                if not gen["text"]:
                    print("    [!] Empty response")
                    continue

                # 1. Save TXT immediately
                txt_file = save_txt(record_id, gen["text"])
                print(f"    TXT saved: {txt_file}")

                # 2. Analyze text
                params = analyze(gen["text"])

                # 3. Build record
                record = {
                    "record_id":        record_id,
                    "model":            model,
                    "config":           cfg["suffix"],
                    "config_label":     cfg["label"],
                    "prompt":           prompt_key,
                    "temperature":      cfg["temperature"],
                    "top_p":            cfg["top_p"],
                    "repeat_penalty":   cfg["repeat_penalty"],
                    "num_predict_used": gen["num_predict_used"],
                    "num_ctx_used":     gen["num_ctx_used"],
                    "tokens_generated": gen["tokens_generated"],
                    "done_reason":      gen["done_reason"],
                    "tokens_per_sec":   gen["tokens_per_sec"],
                    **params,
                    "txt_file":         txt_file,
                    "full_text":        gen["text"],
                }
                records.append(record)
                done_ids.add(record_id)

                # 4. Save JSON + all CSVs after every record
                save_all(records)

                print(f"    chars={params['char_count']:,}  "
                      f"words={params['word_count']:,}  "
                      f"ttr={params['ttr']:.3f}  "
                      f"herdan={params['herdan_c']:.3f}  "
                      f"disc/100={params['discourse_per_100w']:.2f}  "
                      f"tps={gen['tokens_per_sec']:.0f}")

    print_summary(records)

    # Final archive
    print("\n[+] Creating download archive...")
    os.system(
        f"tar -czf diploma_results_v3.tar.gz "
        f"{TEXTS_DIR}/ "
        f"{MAIN_CSV} {MAIN_JSON} "
        f"{MODEL_CSV} {CONFIG_CSV} {PROMPT_CSV} "
        f"2>/dev/null"
    )
    if os.path.exists("diploma_results_v3.tar.gz"):
        mb = os.path.getsize("diploma_results_v3.tar.gz") / 1024 / 1024
        print(f"[+] diploma_results_v3.tar.gz ({mb:.1f} MB)")
        print("[+] Download: Jupyter → File Browser → right click → Download")

    elapsed = time.time() - t0
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\n[+] Done in {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
