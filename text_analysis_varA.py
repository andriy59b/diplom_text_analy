#!/usr/bin/env python3
"""
Варіант А — Генерація до ліміту токенів
Моделі: gemma4, gemma3, granite, phi4, deepseek, mistral-nemo, gpt-oss
Пропущено: qwen2.5:3b і qwen3:14b (вже є)
1 конфіг (Standard) × 1 промт (maxlength)
Andrii Pentsak, LNU Lviv, 2026
"""

import os, sys, re, time, csv, json, platform, subprocess
from pathlib import Path
from typing import List, Dict, Any

try:
    import requests
except ImportError:
    print("pip install requests"); sys.exit(1)

OLLAMA_URL   = "http://localhost:11434/api/generate"
JSON_FILE    = "results_varA.json"
CSV_FILE     = "results_varA.csv"
TEXTS_DIR    = Path("generated_texts_varA")
TEXTS_DIR.mkdir(exist_ok=True)

# ── Моделі (без qwen2.5:3b і qwen3:14b — вже є) ──────────────────────────────
TARGET_MODELS = [
    "gemma4:e4b",
    "gemma3:12b",
    "granite4.1:8b",
    "phi4:14b",
    "deepseek-r1:1.5b",
    "mistral-nemo:12b",
    "gpt-oss:20b",
]

# Ліміти токенів — повільним менше щоб не витрачати весь бюджет
MODEL_LIMITS = {
    "gemma4:e4b":        96000,
    "gemma3:12b":        96000,
    "granite4.1:8b":     30000,  # 18 TPS — повільна
    "phi4:14b":          96000,
    "deepseek-r1:1.5b":  60000,
    "mistral-nemo:12b":  40000,  # 22 TPS — повільна
    "gpt-oss:20b":       96000,
}

# ── Єдиний промт — maxlength з забороною на висновки ─────────────────────────
PROMPT = """You must use ALL available tokens to write the longest possible scientific monograph on "Large Language Models: Architecture, Training, and Applications".

CRITICAL RULES:
- Use every single token — do NOT stop until reaching the absolute token limit
- Do NOT write conclusions, summary, bibliography or references — they signal end of text
- After finishing all chapters immediately start chapter 1 again with MORE detail
- Use synonyms and varied vocabulary — avoid repeating the same words
- Every concept must be explained from multiple angles
- Expand every bullet point into multiple full paragraphs
- Never use the same phrase twice

Chapters to cover exhaustively:
1. Introduction and history of AI development
2. Mathematical foundations of transformer architecture with all formulas
3. LLM training: pre-training, fine-tuning, RLHF with detailed algorithms
4. Modern models: GPT series, BERT family, LLaMA, Mistral, Claude, Gemini, PaLM
5. Text quality evaluation: BLEU, ROUGE, BERTScore, perplexity, human evaluation
6. Real-world applications: medicine, education, programming, law, science
7. Problems: hallucinations, bias, safety, alignment, energy consumption
8. Future: AGI, multimodal models, efficiency improvements

Write continuously without stopping. Maximize output length. Never conclude."""

# Конфігурація генерації
CONFIG = {
    "temperature":    0.9,
    "top_p":          0.95,
    "repeat_penalty": 1.0,
    "num_ctx":        65536,
}

# ── Аналіз тексту ─────────────────────────────────────────────────────────────
DISCOURSE_MARKERS = [
    "however","furthermore","moreover","therefore","consequently",
    "in addition","on the other hand","for example","for instance",
    "in conclusion","thus","hence","nevertheless","nonetheless",
    "in contrast","similarly","specifically","notably","importantly",
]

def analyze(text: str) -> Dict:
    import math, statistics as st
    words = text.split()
    lwords = text.lower().split()
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    sent_lens = [len(s.split()) for s in sents]
    unique = len(set(lwords))
    n = len(lwords)
    ttr = round(unique/n, 4) if n > 0 else 0
    herdan = round(math.log(unique)/math.log(n), 4) if n > 1 and unique > 1 else 0
    return {
        "char_count":          len(text),
        "word_count":          len(words),
        "sentence_count":      len(sents),
        "paragraph_count":     len(paras),
        "unique_words":        unique,
        "ttr":                 ttr,
        "herdan_c":            herdan,
        "avg_sentence_len":    round(st.mean(sent_lens),2) if sent_lens else 0,
        "std_sentence_len":    round(st.stdev(sent_lens),2) if len(sent_lens)>1 else 0,
        "discourse_markers":   sum(text.lower().count(m) for m in DISCOURSE_MARKERS),
        "discourse_per_100w":  round(sum(text.lower().count(m) for m in DISCOURSE_MARKERS)/len(words)*100,2) if words else 0,
    }

# ── Генерація ─────────────────────────────────────────────────────────────────
def generate(model: str, num_predict: int) -> Dict:
    full_prompt = (
        "# Large Language Models: Architecture, Training, and Applications\n\n"
        "## Chapter 1: Introduction and Historical Development\n\n"
        + PROMPT
    )
    payload = {
        "model":  model,
        "prompt": full_prompt,
        "stream": True,
        "raw":    True,
        "options": {
            **CONFIG,
            "num_predict": num_predict,
            "stop": [],  # пустий список — жодного стоп-токена
        },
    }
    text = ""
    tokens = 0
    done_reason = "unknown"
    eval_ns = 0
    t0 = time.time()

    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=28800)
        for line in resp.iter_lines():
            if not line: continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    text   += chunk["response"]
                    tokens += 1
                    if tokens % 2000 == 0:
                        elapsed = time.time() - t0
                        tps_cur = tokens / elapsed if elapsed > 0 else 0
                        print(f"    {tokens:,} токенів / {len(text):,} символів / {tps_cur:.0f} t/s")
                if chunk.get("done"):
                    done_reason = chunk.get("done_reason", "unknown")
                    eval_ns     = chunk.get("eval_duration", 0)
                    print(f"    Зупинка: {done_reason} @ {tokens:,} токенів / {len(text):,} символів")
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"    Помилка: {e}")

    tps = tokens / (eval_ns/1e9) if eval_ns > 0 else 0
    return {
        "text":             text,
        "tokens_generated": tokens,
        "done_reason":      done_reason,
        "tokens_per_sec":   round(tps, 1),
    }

# ── Збереження ────────────────────────────────────────────────────────────────
def load_results() -> List[Dict]:
    try:
        with open(JSON_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Завантажено {len(data)} існуючих записів")
        return data
    except FileNotFoundError:
        return []

def save_results(results: List[Dict]) -> None:
    slim = [{k:v for k,v in r.items() if k != "full_text"} for r in results]
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    fields = [
        "record_id","model","temperature","top_p","repeat_penalty",
        "num_predict","num_ctx","char_count","word_count","sentence_count",
        "paragraph_count","unique_words","ttr","herdan_c","avg_sentence_len",
        "std_sentence_len","discourse_markers","discourse_per_100w",
        "tokens_generated","done_reason","tokens_per_sec","txt_file",
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

def save_text(record_id: str, text: str) -> str:
    safe = record_id.replace(":", "-").replace("/", "-")
    path = TEXTS_DIR / f"{safe}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)

def check_models() -> List[str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        available = [m["name"] for m in resp.json().get("models", [])]
        active = []
        print("[+] Доступність моделей:")
        for m in TARGET_MODELS:
            found = any(a == m or a.startswith(m.split(":")[0]) for a in available)
            print(f"    {'✓' if found else '✗ ВІДСУТНЯ'} {m}  (ліміт: {MODEL_LIMITS.get(m,96000):,} токенів)")
            if found:
                active.append(m)
        return active
    except Exception as e:
        print(f"[-] Ollama недоступна: {e}")
        return []

def print_summary(results: List[Dict]) -> None:
    valid = [r for r in results if r.get("char_count",0) > 0]
    print("\n" + "="*90)
    print("  ПІДСУМОК РЕЗУЛЬТАТІВ")
    print("="*90)
    print(f"{'Модель':<25} {'Символів':>10} {'Слів':>8} {'TTR':>6} {'Херд.C':>7} {'TPS':>7}  Стоп")
    print("-"*90)
    for r in valid:
        print(f"{r['model']:<25} {r['char_count']:>10,} {r['word_count']:>8,} "
              f"{r['ttr']:>6.3f} {r['herdan_c']:>7.3f} {r['tokens_per_sec']:>7.1f}  "
              f"{r.get('done_reason','?')}")
    print("="*90)
    total = sum(r['char_count'] for r in valid)
    length_n = sum(1 for r in valid if r.get('done_reason')=='length')
    print(f"\nЗаписів:      {len(valid)}")
    print(f"Всього символів: {total:,}")
    print(f"До ліміту (length): {length_n}/{len(valid)}")

# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*65)
    print("  ВАРІАНТ А — ГЕНЕРАЦІЯ ДО ЛІМІТУ ТОКЕНІВ")
    print("  Промт: maxlength | Конфіг: Standard")
    print(f"  Host: {platform.node()}")
    print("="*65)

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],
            encoding="utf-8").strip()
        print(f"[+] GPU: {gpu}")
    except Exception:
        pass

    active = check_models()
    if not active:
        print("[-] Немає доступних моделей")
        return

    print(f"\n[+] Моделей для генерації: {len(active)}")
    print(f"[+] Тексти зберігаються в: {TEXTS_DIR}/\n")

    results  = load_results()
    done_ids = {r["record_id"] for r in results}

    for model in active:
        num_predict = MODEL_LIMITS.get(model, 96000)
        record_id   = f"{model}__std__maxlength"

        if record_id in done_ids:
            print(f"\n[~] Пропускаємо {model} — вже є")
            continue

        print(f"\n{'='*65}")
        print(f"  МОДЕЛЬ: {model}")
        print(f"  Ліміт: {num_predict:,} токенів | T={CONFIG['temperature']} | ctx={CONFIG['num_ctx']}")
        print(f"{'='*65}")

        gen = generate(model, num_predict)

        if not gen["text"]:
            print("  [!] Порожня відповідь — пропускаємо")
            continue

        # Зберігаємо текст одразу
        txt_path = save_text(record_id, gen["text"])
        print(f"  Текст збережено: {txt_path}")

        params = analyze(gen["text"])

        record = {
            "record_id":        record_id,
            "model":            model,
            "prompt":           "prompt_maxlength",
            "config":           "std",
            "temperature":      CONFIG["temperature"],
            "top_p":            CONFIG["top_p"],
            "repeat_penalty":   CONFIG["repeat_penalty"],
            "num_predict":      num_predict,
            "num_ctx":          CONFIG["num_ctx"],
            **params,
            "tokens_generated": gen["tokens_generated"],
            "done_reason":      gen["done_reason"],
            "tokens_per_sec":   gen["tokens_per_sec"],
            "txt_file":         txt_path,
            "full_text":        gen["text"],
        }
        results.append(record)
        done_ids.add(record_id)

        print(f"\n  Символів:     {params['char_count']:,}")
        print(f"  Слів:         {params['word_count']:,}")
        print(f"  TTR:          {params['ttr']:.4f}")
        print(f"  Херданс C:    {params['herdan_c']:.4f}")
        print(f"  Дискурс/100:  {params['discourse_per_100w']:.2f}")
        print(f"  TPS:          {gen['tokens_per_sec']:.1f}")
        print(f"  Зупинка:      {gen['done_reason']}")

        save_results(results)
        print(f"  Збережено в {JSON_FILE}")

    print_summary(results)

    # Архів для скачування
    print("\n[+] Створюємо архів...")
    os.system(f"tar -czf varA_results.tar.gz {TEXTS_DIR}/ {JSON_FILE} {CSV_FILE} 2>/dev/null")
    if os.path.exists("varA_results.tar.gz"):
        mb = os.path.getsize("varA_results.tar.gz") / 1024 / 1024
        print(f"[+] varA_results.tar.gz ({mb:.1f} MB)")
        print("[+] Скачай: Jupyter → File Browser → правий клік → Download")

    elapsed = time.time() - t0
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\n[+] Готово за {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
