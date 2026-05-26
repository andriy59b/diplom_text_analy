#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
OLLAMA LLM PERFORMANCE AND INFERENCE BENCHMARKING SUITE
Extended for Bachelor's Thesis:
"Analysis of Output Text Parameters in Modern Generative Language Models"
Andrii Pentsak, Ivan Franko National University of Lviv, 2026

Original benchmark architecture:
  t3dr0/ollama_benchmark (AGPLv3)
  "Reusing Obsolete Windows 10 PCs for On-Premises LLM Inference"
  I. Curington and K. Lano (2026)

Extensions added for thesis research:
  - Three Modelfile configurations per model (standard / creative / precise)
    targeting 35,000–50,000 tokens per response via Ollama create API
  - Three distinct long-generation prompts per configuration
  - Text parameter analysis: TTR, avg sentence length, discourse markers,
    char/word/sentence/paragraph counts, done_reason, tokens_per_sec
  - Extended CSV and JSON output with all original + new metrics
  - Full generation text saved per record for corpus analysis

Features (original):
  - Automated System Diagnostics: Hostname, OS, GPU/VRAM info
  - VRAM Warming: cold-start warm-up for consistent measurements
  - Diverse Prompt Suite: 10 benchmarks (math, logic, code, ethics)
  - Long-Context Stress Test: 2,200+ word KV cache payload
  - Statistical Analysis: Mean, Variance, Std Dev over N runs
  - Multi-Turn Tracking: performance at T1, T3, T5
  - Structured Reporting: console report + CSV

Version: 1.1.0
License: AGPLv3
--------------------------------------------------------------------------------
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
from typing import List, Dict, Any, Tuple, Optional, Union

# ── Network patch (original) ─────────────────────────────────────────────────
ollama_host: str = os.environ.get("OLLAMA_HOST", "")
if "0.0.0.0" in ollama_host:
    print("\n[!] Network Notice: OLLAMA_HOST is targeting '0.0.0.0'.")
    print("    Non-routable for client requests. Patching to loopback...")
    port: str = ollama_host.split(":")[-1] if ":" in ollama_host else "11434"
    patched_host: str = f"http://127.0.0.1:{port}"
    os.environ["OLLAMA_HOST"] = patched_host
    print(f"    -> Re-routed to: {patched_host}\n")

try:
    import ollama
except ImportError:
    print("[-] Error: pip install ollama")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[-] Error: pip install requests")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 1: ORIGINAL BENCHMARK CONFIGURATION (unchanged) ──────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Original 8 benchmark models
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

NUM_RUNS: int = 5
CSV_FILENAME: str = "ollama_benchmark_results.csv"

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 2: THESIS EXTENSION CONFIGURATION ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Output files for text analysis extension
TEXT_CSV_FILENAME:  str = "text_analysis_results.csv"
TEXT_JSON_FILENAME: str = "text_analysis_results.json"

# Three generation configurations — each model gets 3 custom variants
# Named: <base_model>-std, <base_model>-creative, <base_model>-precise
TEXT_CONFIGS: List[Dict[str, Any]] = [
    {
        "suffix":      "std",
        "label":       "Standard",
        "num_predict": 40000,
        "temperature": 0.9,
        "top_p":       0.95,
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
        "suffix":      "creative",
        "label":       "Creative",
        "num_predict": 50000,
        "temperature": 1.1,
        "top_p":       0.98,
        "repeat_penalty": 1.0,
        "system": (
            "You are a scientific writing assistant with a creative flair. "
            "Write the longest possible text using diverse vocabulary and varied sentence structures. "
            "Use synonyms extensively — never repeat the same phrase twice. "
            "Expand every concept from multiple angles with different terminology. "
            "Never stop — keep writing until all tokens are exhausted."
        ),
    },
    {
        "suffix":      "precise",
        "label":       "Precise",
        "num_predict": 35000,
        "temperature": 0.7,
        "top_p":       0.90,
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

# Three long-generation prompts
TEXT_PROMPTS: Dict[str, str] = {
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
        "2. Mathematical foundations of transformer architecture (all formulas step by step)\n"
        "3. LLM training: pre-training, fine-tuning, RLHF with detailed algorithms\n"
        "4. Modern models: GPT series, BERT family, LLaMA versions, Mistral, Claude, Gemini, PaLM\n"
        "5. Evaluation: BLEU, ROUGE, BERTScore, perplexity, human evaluation protocols\n"
        "6. Applications: medicine, education, programming, law, creative writing, science\n"
        "7. Problems: hallucinations, bias, safety, alignment, energy consumption\n"
        "8. Future: AGI, multimodal models, efficiency improvements\n\n"
        "Write continuously without stopping. Maximize output length."
    ),
    "prompt_detailed": (
        "Act as a world-class professor writing the most detailed academic textbook ever "
        "written on Large Language Models.\n\n"
        'Your textbook "Large Language Models: A Complete Scientific Reference" '
        "must be exhaustive and encyclopedic.\n\n"
        "For EACH chapter write AT MINIMUM 10 full pages of dense academic content:\n\n"
        "Chapter 1 — Historical Foundation: Every milestone from McCulloch-Pitts neuron (1943) "
        "through perceptrons, backpropagation, attention mechanisms to modern transformers. "
        "Include dates, researchers, institutions, breakthroughs and failures.\n\n"
        "Chapter 2 — Mathematical Foundations: Derive every formula from first principles. "
        "Cover linear algebra, probability theory, information theory, "
        "optimization theory, attention mechanism mathematics, positional encoding derivations.\n\n"
        "Chapter 3 — Training Methodology: BPE, WordPiece, SentencePiece tokenization, "
        "pre-training objectives, fine-tuning strategies, RLHF pipeline, "
        "Constitutional AI, instruction tuning with complete technical detail.\n\n"
        "Chapter 4 — Model Architectures: Compare encoder-only (BERT), decoder-only (GPT), "
        "encoder-decoder (T5). Analyze GPT-1 through GPT-4, LLaMA 1/2/3, Mistral 7B/8x7B, "
        "Claude 1/2/3, Gemini, PaLM, Falcon with parameter counts, training data, benchmarks.\n\n"
        "Chapter 5 — Evaluation Framework: BLEU with mathematical derivation, ROUGE variants, "
        "BERTScore computation, perplexity calculation, human evaluation methodologies, "
        "benchmark suites (MMLU, HellaSwag, TruthfulQA).\n\n"
        "Chapter 6 — Applications: Real deployments in healthcare diagnostics, legal document "
        "analysis, code generation, scientific research, education personalization "
        "with specific case studies and measured outcomes.\n\n"
        "Chapter 7 — Challenges: Hallucination mechanisms, bias sources and mitigation, "
        "adversarial attacks, privacy concerns, environmental impact "
        "with specific statistics and research findings.\n\n"
        "Chapter 8 — Future Directions: Scaling laws, emergent abilities, multimodal integration, "
        "efficiency research, AGI implications.\n\n"
        "Write every word of every chapter. Do not abbreviate. Do not skip sections. "
        "Continue writing until all tokens are exhausted. "
        "Do NOT write a bibliography or references section."
    ),
}

# Discourse markers for text analysis
DISCOURSE_MARKERS: List[str] = [
    "however", "furthermore", "moreover", "therefore", "consequently",
    "in addition", "on the other hand", "for example", "for instance",
    "in conclusion", "thus", "hence", "nevertheless", "nonetheless",
    "in contrast", "similarly", "specifically", "notably", "importantly",
]

# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 3: ORIGINAL HELPER FUNCTIONS (unchanged) ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_gpu_info() -> str:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpus: List[str] = []
        for line in res.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                gpus.append(f"{parts[0].strip()} ({parts[1].strip()} MB VRAM)")
        return ", ".join(gpus)
    except Exception:
        return "Unknown / Non-Nvidia GPU (Could not parse nvidia-smi)"


def generate_long_text() -> str:
    paragraphs: List[str] = [
        "The institutional landscape of the modern global economy has quietly transitioned from legacy framework operations to highly integrated, algorithmically-driven infrastructure nodes. Over the past decade, the rapid scaling of distributed cloud computing architectures and neural networks has transformed how enterprises process vast datasets, handle logistics, and orchestrate consumer touchpoints. This systematic migration towards automation is not merely an optimization of existing practices; it represents a fundamental paradigm shift in organizational sociology.",
        "Simultaneously, the displacement patterns of modern automation have broken historical precedents. While twentieth-century industrial automation predominantly targeted routine, manual blue-collar tasks, the current wave of cognitive automation is directly encroaching upon non-routine, analytical white-collar domains. Content creation, legal discovery, contract analysis, financial forecasting, and even software compilation are no longer insulated from machine intelligence. Large language architectures and specialized transformers possess the capability to synthesize legal precedents across thousands of pages within seconds.",
        "Beyond the structural economic reallocations, the psychological impact of operating within these highly optimized digital environments remains profoundly complex. Modern workers are subject to unprecedented levels of telemetry and behavioral tracking. Every keystroke, mouse movement, communication interval, and task completion metric is continuously logged and processed by internal performance management algorithms. This totalizing panoptic surveillance creates a culture of hyper-surveillance that reshapes employee psychology.",
        "Furthermore, the ethical governance of these autonomous frameworks remains a critical vulnerability. As models assume greater responsibility over resource allocation, credit scoring, and algorithmic hiring, algorithmic biases become deeply entrenched within institutional infrastructure. These biases, often reflective of historical inequities hidden within the training datasets, are processed as objective mathematical truths by automated nodes. Without continuous intervention, rigorous algorithmic auditing, and human-in-the-loop oversight, these systems risk replicating and amplifying systemic inequality at a speed and scale previously unimaginable.",
    ]
    full_text: List[str] = []
    while sum(len(p.split()) for p in full_text) < 2300:
        full_text.extend(paragraphs)
    return "\n\n".join(full_text)


def extract_metric(response: Any, key: str) -> Optional[Any]:
    if hasattr(response, key):
        return getattr(response, key)
    elif isinstance(response, dict):
        return response.get(key)
    elif hasattr(response, "model_dump"):
        return response.model_dump().get(key)
    return None


def run_environment_checks() -> Tuple[str, str, List[str], Dict[str, str]]:
    print("=" * 60)
    print(" SYSTEM DIAGNOSTICS & INITIALIZATION")
    print("=" * 60)
    hostname: str = platform.node()
    os_info:  str = platform.platform()
    gpu_info: str = get_gpu_info()

    print(f"[+] Hostname: {hostname}")
    print(f"[+] OS Context: {os_info}")
    print(f"[+] Hardware Accelerator: {gpu_info}")

    try:
        local_manifest: Any = ollama.list()
        model_size_map: Dict[str, str] = {}
        if hasattr(local_manifest, "models"):
            for m in local_manifest.models:
                gb_size = getattr(m, "size", 0) / (1000 ** 3)
                model_size_map[m.model] = f"{gb_size:.1f} GB"
        elif isinstance(local_manifest, dict) and "models" in local_manifest:
            for m in local_manifest["models"]:
                if isinstance(m, dict):
                    model_size_map[m.get("name", "")] = f"{m.get('size', 0) / 1e9:.1f} GB"
        available_models: List[str] = list(model_size_map.keys())
    except Exception as e:
        print(f"\n[-] Error: Failed to communicate with Ollama API. ({e})")
        sys.exit(1)

    print("\n[+] Checking Target Model Availability:")
    active_models:  List[str] = []
    missing_models: List[str] = []
    for model in TARGET_MODELS:
        match = [m for m in available_models if m == model or m.startswith(model + ":")]
        if match:
            print(f"    --> [AVAILABLE] {model}")
            active_models.append(model)
        else:
            print(f"    --> [MISSING]   {model}")
            missing_models.append(model)

    if missing_models:
        print("\n[!] Warning: Some models are missing. Run: ollama pull <model>")
    if not active_models:
        print("\n[-] No models available. Aborting.")
        sys.exit(1)

    print(f"\n[+] Proceeding with {len(active_models)} model(s)...\n")
    return hostname, gpu_info, active_models, model_size_map


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 4: TEXT ANALYSIS FUNCTIONS (thesis extension) ────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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

def count_discourse_markers_fn(text: str) -> int:
    t = text.lower()
    return sum(t.count(m) for m in DISCOURSE_MARKERS)

def analyze_text(text: str) -> Dict[str, Any]:
    return {
        "char_count":        len(text),
        "word_count":        count_words(text),
        "sentence_count":    count_sentences(text),
        "paragraph_count":   count_paragraphs(text),
        "ttr":               type_token_ratio(text),
        "avg_sentence_len":  avg_sentence_length(text),
        "discourse_markers": count_discourse_markers_fn(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 5: MODELFILE CREATION (thesis extension) ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def create_custom_model(base_model: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Create a custom Ollama model with long-generation parameters via Modelfile.
    Returns the custom model name or None if creation failed.
    """
    # Sanitize base model name for use in custom model name
    safe_name = base_model.replace(":", "-").replace(".", "-")
    custom_name = f"{safe_name}-{config['suffix']}"

    modelfile_content = f"""FROM {base_model}

PARAMETER num_predict {config['num_predict']}
PARAMETER temperature {config['temperature']}
PARAMETER top_p {config['top_p']}
PARAMETER repeat_penalty {config['repeat_penalty']}
PARAMETER num_ctx 131072

SYSTEM \"\"\"{config['system']}\"\"\"
"""
    try:
        print(f"    [~] Creating custom model: {custom_name} ...", end=" ", flush=True)
        ollama.create(model=custom_name, modelfile=modelfile_content)
        print("OK")
        return custom_name
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def create_all_custom_models(active_models: List[str]) -> Dict[str, List[str]]:
    """
    Create 3 custom model variants for every active base model.
    Returns a dict: base_model -> [custom_model_std, custom_model_creative, custom_model_precise]
    """
    print("\n" + "=" * 65)
    print("  CREATING CUSTOM LONG-GENERATION MODELS (THESIS EXTENSION)")
    print("=" * 65)
    print("  Each base model gets 3 configurations:")
    print("  - std      : temperature=0.9, num_predict=40000")
    print("  - creative : temperature=1.1, num_predict=50000")
    print("  - precise  : temperature=0.7, num_predict=35000")
    print()

    custom_model_map: Dict[str, List[str]] = {}

    for base_model in active_models:
        print(f"\n  Base model: {base_model}")
        variants: List[str] = []
        for cfg in TEXT_CONFIGS:
            name = create_custom_model(base_model, cfg)
            if name:
                variants.append(name)
        custom_model_map[base_model] = variants
        print(f"  Created: {variants}")

    return custom_model_map


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 6: LONG-GENERATION QUERY (thesis extension) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def query_long_generation(
    custom_model: str,
    prompt_key: str,
    prompt_text: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Query a custom model for long text generation using streaming mode.
    Returns generated text + performance metadata.
    """
    full_prompt = (
        "# Large Language Models: Architecture, Training, and Applications\n\n"
        "## Chapter 1: Introduction and Historical Development\n\n"
        + prompt_text
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model":  custom_model,
        "prompt": full_prompt,
        "stream": True,
        "raw":    False,  # use system prompt from Modelfile
        "options": {
            "num_predict":    config["num_predict"],
            "temperature":    config["temperature"],
            "top_p":          config["top_p"],
            "repeat_penalty": config["repeat_penalty"],
            "num_ctx":        131072,
        },
    }

    full_text   = ""
    token_count = 0
    done_reason = "unknown"
    eval_dur_ns = 0

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=28800)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_text   += chunk["response"]
                    token_count += 1
                    if token_count % 1000 == 0:
                        print(f"      ... {token_count:,} tokens / {len(full_text):,} chars")
                if chunk.get("done", False):
                    done_reason = chunk.get("done_reason", "unknown")
                    eval_dur_ns = chunk.get("eval_duration", 0)
                    print(f"      Stopped: {done_reason} at {token_count:,} tokens")
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"      [!] Error: {e}")

    tps = (token_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

    return {
        "text":             full_text,
        "tokens_generated": token_count,
        "done_reason":      done_reason,
        "tokens_per_sec":   round(tps, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 7: TEXT ANALYSIS RESULTS PERSISTENCE ─────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_text_results() -> List[Dict[str, Any]]:
    try:
        with open(TEXT_JSON_FILENAME, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Loaded {len(data)} existing text analysis results")
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[!] Could not load text results: {e}")
        return []


def save_text_results(results: List[Dict[str, Any]]) -> None:
    # Full data with text → JSON
    with open(TEXT_JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary without full_text → CSV
    fields = [
        "config", "base_model", "custom_model", "config_label",
        "prompt_key", "temperature", "top_p", "repeat_penalty", "num_predict",
        "char_count", "word_count", "sentence_count", "paragraph_count",
        "ttr", "avg_sentence_len", "discourse_markers",
        "tokens_generated", "done_reason", "tokens_per_sec",
    ]
    with open(TEXT_CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 8: ORIGINAL BENCHMARK FUNCTIONS (unchanged) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_original_benchmark(
    active_models:  List[str],
    model_size_map: Dict[str, str],
    hostname:       str,
    gpu_info:       str,
) -> None:
    """Run the original 10-prompt benchmark suite (unchanged from t3dr0/ollama_benchmark)."""

    long_text_payload: str = generate_long_text()

    prompts_suite: Dict[str, Dict[str, str]] = {
        "P1":  {"name": "Short Factual Retrieval",
                "text": "Name the SI unit of electrical resistance and state its symbol."},
        "P2":  {"name": "Long Structured Prose",
                "text": "Explain the causes and consequences of the 2008 financial crisis in 1,000 words."},
        "P3":  {"name": "Multi-Step Arithmetic Reasoning",
                "text": "A train travels from A to B at 80 km/h and returns at 120 km/h. "
                        "If the total journey takes 5 hours, what is the distance between A and B? "
                        "Show all working."},
        "P4":  {"name": "Code Generation",
                "text": "Write a Python function that accepts a list of integers and returns "
                        "a dictionary mapping each unique value to the list of indices at which "
                        "it appears. Include docstring and type hints."},
        "P5":  {"name": "Long-Context Summarisation",
                "text": f"Summarise the following text in five bullet points, "
                        f"each no longer than 30 words.\n\n{long_text_payload}"},
        "P6":  {"name": "Instruction Following Constraints",
                "text": "Write a product description for a mechanical keyboard. "
                        "Requirements: exactly 3 paragraphs, no use of the word 'perfect', "
                        "final paragraph must include a call to action, reading age below 12."},
        "P7":  {"name": "Logical/Deductive Reasoning",
                "text": "All members of the board approved the motion. "
                        "Sarah did not approve the motion. Is Sarah a board member? "
                        "Explain your reasoning step by step."},
        "P8":  {"name": "Multilingual Translation Matrix",
                "text": "Translate the following paragraph into formal French, then "
                        "back-translate your French into English, and identify any meaning "
                        "lost in the process.\n\nParagraph: The rapid advancement of artificial "
                        "intelligence has sparked a global debate on safety and ethics."},
        "P10": {"name": "Refusal / Edge Case Robustness",
                "text": "Explain how social engineering attacks work, and describe three "
                        "techniques a security awareness trainer might use to demonstrate "
                        "them to employees."},
    }

    p9_turns: List[str] = [
        "Hello! I am planning a research project on the history of printing presses. "
        "Can you give me a brief overview of Johannes Gutenberg's contribution?",
        "Fascinating. What were the primary social and economic impacts of this "
        "technological shift in 15th-century Europe?",
        "How did the Catholic Church react to the sudden proliferation of printed "
        "materials during that period?",
        "Can you compare this historical information explosion to the rise of the "
        "early internet in the late 20th century?",
        "Summarize the key parallels you just drew between the printing press and "
        "the internet into three concise bullet points.",
    ]

    warmup_prompt: str = (
        "Write a simple Python function that calculates the factorial of a given integer. "
        "Omit explanations, bare code only."
    )

    results_records: List[Dict[str, Any]] = []

    for model in active_models:
        print("=" * 70)
        print(f" BENCHMARKING MODEL: {model}")
        print("=" * 70)

        print("[~] Warm-up sequence...")
        try:
            ollama.chat(model=model, messages=[{"role": "user", "content": warmup_prompt}])
            print("[+] Warm-up complete.\n")
        except Exception as e:
            print(f"[-] Warm-up failed for {model}: {e}. Skipping.")
            continue

        model_size_str = model_size_map.get(model, "Unknown")

        for pid, pdata in prompts_suite.items():
            print(f"  Running {pid}: {pdata['name']} ({NUM_RUNS}x)...")
            tps_runs: List[float]     = []
            elapsed_runs: List[float] = []

            for _ in range(NUM_RUNS):
                try:
                    response: Any = ollama.chat(
                        model=model,
                        messages=[{"role": "user", "content": pdata["text"]}]
                    )
                    eval_count       = extract_metric(response, "eval_count") or 0
                    eval_duration_ns = extract_metric(response, "eval_duration") or 0
                    total_duration_ns= extract_metric(response, "total_duration") or 0
                    tps     = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
                    elapsed = total_duration_ns / 1e9 if total_duration_ns > 0 else 0.0
                    tps_runs.append(tps)
                    elapsed_runs.append(elapsed)
                except Exception:
                    tps_runs.append(0.0)
                    elapsed_runs.append(0.0)

            for metric, vals in [("Tokens_Per_Sec", tps_runs), ("Elapsed_Sec", elapsed_runs)]:
                results_records.append({
                    "model": model, "size": model_size_str,
                    "pid": pid, "pname": pdata["name"], "metric": metric,
                    "runs": vals,
                    "mean":     statistics.mean(vals),
                    "variance": statistics.variance(vals) if len(vals) > 1 else 0,
                    "stdev":    statistics.stdev(vals)    if len(vals) > 1 else 0,
                })

        # P9 Multi-Turn
        print(f"  Running P9: Multi-Turn Conversation ({NUM_RUNS}x)...")
        p9_tracking: Dict[int, Dict[str, List[float]]] = {
            1: {"tps": [], "elap": []},
            3: {"tps": [], "elap": []},
            5: {"tps": [], "elap": []},
        }
        for _ in range(NUM_RUNS):
            messages: List[Dict[str, str]] = []
            for turn_idx, user_content in enumerate(p9_turns, 1):
                messages.append({"role": "user", "content": user_content})
                try:
                    resp: Any = ollama.chat(model=model, messages=messages)
                    assist = extract_metric(resp, "message")
                    messages.append({
                        "role": "assistant",
                        "content": (
                            assist.get("content", "")
                            if isinstance(assist, dict)
                            else getattr(assist, "content", "")
                        ),
                    })
                    if turn_idx in [1, 3, 5]:
                        t_count = extract_metric(resp, "eval_count") or 0
                        t_dur   = extract_metric(resp, "eval_duration") or 0
                        p9_tracking[turn_idx]["tps"].append(
                            t_count / (t_dur / 1e9) if t_dur > 0 else 0.0
                        )
                        p9_tracking[turn_idx]["elap"].append(
                            (extract_metric(resp, "total_duration") or 0) / 1e9
                        )
                except Exception:
                    break

        for turn in [1, 3, 5]:
            tps_v  = p9_tracking[turn]["tps"]
            elap_v = p9_tracking[turn]["elap"]
            for m_type, vals in [("Tokens_Per_Sec", tps_v), ("Elapsed_Sec", elap_v)]:
                results_records.append({
                    "model": model, "size": model_size_str,
                    "pid": f"P9_T{turn}", "pname": f"Multi-Turn Dialogue (Turn {turn})",
                    "metric": m_type, "runs": vals,
                    "mean":     statistics.mean(vals) if vals else 0,
                    "variance": statistics.variance(vals) if len(vals) > 1 else 0,
                    "stdev":    statistics.stdev(vals)    if len(vals) > 1 else 0,
                })

    # CSV export (original format)
    try:
        with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# OLLAMA LOCAL INFERENCE BENCHMARK REPORT"])
            writer.writerow([
                f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Host: {hostname}",
                f"# GPU: {gpu_info}",
            ])
            writer.writerow([
                "Model", "Size", "Prompt_ID", "Prompt_Name", "Metric_Type",
                "Run_1", "Run_2", "Run_3", "Run_4", "Run_5",
                "Mean", "Variance", "Std_Dev",
            ])
            for r in results_records:
                pad = (r["runs"] + [0.0] * 5)[:5]
                writer.writerow([
                    r["model"], r["size"], r["pid"], r["pname"], r["metric"],
                ] + pad + [
                    f"{r['mean']:.4f}", f"{r['variance']:.4f}", f"{r['stdev']:.4f}",
                ])
        print(f"\n[+] Original benchmark results saved to: {CSV_FILENAME}")
    except IOError as e:
        print(f"\n[-] Error saving original CSV: {e}")

    # Console summary (original format)
    print("\n" + "=" * 85)
    print(" SUMMARY BENCHMARK OUTPUT ENGINE REPORT  (original suite)")
    print("=" * 85)
    print(f"{'Model':<22} | {'ID':<7} | {'Metric':<25} | {'Mean':>12} | {'StdDev':>8}")
    print("-" * 85)
    for r in results_records:
        unit = " t/sec" if r["metric"] == "Tokens_Per_Sec" else " sec  "
        print(
            f"{r['model']:<22} | {r['pid']:<7} | {r['metric']:<25} | "
            f"{r['mean']:>7.2f}{unit} | {r['stdev']:>8.2f}"
        )
    print("=" * 85)


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION 9: THESIS TEXT ANALYSIS BENCHMARK ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_text_analysis_benchmark(
    active_models:    List[str],
    custom_model_map: Dict[str, List[str]],
) -> None:
    """
    For each base model × 3 custom configs × 3 prompts:
    generate long text and compute text parameters.
    Total: 8 models × 3 configs × 3 prompts = 72 records.
    """
    print("\n" + "=" * 65)
    print("  TEXT PARAMETERS ANALYSIS BENCHMARK  (thesis extension)")
    print("=" * 65)
    print("  Matrix: 8 models × 3 configs × 3 prompts = 72 records")
    print()

    results = load_text_results()
    done = {r["config"] for r in results}

    for base_model in active_models:
        custom_variants = custom_model_map.get(base_model, [])
        if not custom_variants:
            print(f"  [!] No custom models for {base_model} — skipping text analysis")
            continue

        for cfg, custom_model in zip(TEXT_CONFIGS, custom_variants):
            for prompt_key, prompt_text in TEXT_PROMPTS.items():
                config_id = f"{custom_model}__{prompt_key}"

                if config_id in done:
                    print(f"  [~] Skip {config_id} (already done)")
                    continue

                print(f"\n  {'─'*60}")
                print(f"  Base:   {base_model}")
                print(f"  Config: {cfg['label']} ({cfg['suffix']}) | T={cfg['temperature']} | "
                      f"top_p={cfg['top_p']} | max_tokens={cfg['num_predict']}")
                print(f"  Prompt: {prompt_key}")
                print(f"  {'─'*60}")

                gen = query_long_generation(custom_model, prompt_key, prompt_text, cfg)

                if not gen["text"]:
                    print("  [!] Empty response — skipping")
                    continue

                params = analyze_text(gen["text"])

                record: Dict[str, Any] = {
                    "config":          config_id,
                    "base_model":      base_model,
                    "custom_model":    custom_model,
                    "config_label":    cfg["label"],
                    "prompt_key":      prompt_key,
                    "temperature":     cfg["temperature"],
                    "top_p":           cfg["top_p"],
                    "repeat_penalty":  cfg["repeat_penalty"],
                    "num_predict":     cfg["num_predict"],
                    **params,
                    "tokens_generated": gen["tokens_generated"],
                    "done_reason":      gen["done_reason"],
                    "tokens_per_sec":   gen["tokens_per_sec"],
                    "full_text":        gen["text"],
                }
                results.append(record)
                done.add(config_id)

                print(f"  Characters:      {params['char_count']:,}")
                print(f"  Words:           {params['word_count']:,}")
                print(f"  Sentences:       {params['sentence_count']:,}")
                print(f"  Paragraphs:      {params['paragraph_count']:,}")
                print(f"  TTR:             {params['ttr']:.4f}")
                print(f"  Avg sent len:    {params['avg_sentence_len']:.1f} words")
                print(f"  Discourse marks: {params['discourse_markers']}")
                print(f"  Tokens/sec:      {gen['tokens_per_sec']:.1f}")
                print(f"  Stop reason:     {gen['done_reason']}")

                # Save after every record — safe against crashes
                save_text_results(results)

    # ── Text analysis summary ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  TEXT PARAMETERS SUMMARY")
    print("=" * 110)
    print(
        f"{'Config':<48} | {'Chars':>9} | {'Words':>7} | "
        f"{'TTR':>6} | {'AvgSen':>6} | {'Disc':>5} | "
        f"{'TPS':>7} | Stop"
    )
    print("-" * 110)
    for r in results:
        if not r.get("char_count"):
            continue
        print(
            f"{r['config'][:48]:<48} | "
            f"{r['char_count']:>9,} | "
            f"{r['word_count']:>7,} | "
            f"{r['ttr']:>6.3f} | "
            f"{r['avg_sentence_len']:>6.1f} | "
            f"{r['discourse_markers']:>5} | "
            f"{r['tokens_per_sec']:>7.1f} | "
            f"{r.get('done_reason', '?')}"
        )
    print("=" * 110)
    print(f"\n[+] Text analysis results → {TEXT_JSON_FILENAME} + {TEXT_CSV_FILENAME}")


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    start_bench_time = time.time()

    # ── Phase 1: System check (original) ─────────────────────────────────────
    hostname, gpu_info, active_models, model_size_map = run_environment_checks()

    # ── Phase 2: Create custom long-generation models (thesis extension) ──────
    custom_model_map = create_all_custom_models(active_models)

    # ── Phase 3: Original benchmark suite ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 3: RUNNING ORIGINAL BENCHMARK SUITE")
    print("=" * 65)
    run_original_benchmark(active_models, model_size_map, hostname, gpu_info)

    # ── Phase 4: Text analysis benchmark (thesis extension) ───────────────────
    print("\n" + "=" * 65)
    print("  PHASE 4: RUNNING TEXT PARAMETERS ANALYSIS")
    print("=" * 65)
    run_text_analysis_benchmark(active_models, custom_model_map)

    # ── Final timing ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_bench_time
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\n[+] Total execution time: {h:02d}:{m:02d}:{s:02d}")
    print(f"[+] Output files:")
    print(f"    {CSV_FILENAME}       — original benchmark (TPS, latency)")
    print(f"    {TEXT_CSV_FILENAME}  — text analysis (TTR, discourse markers, etc.)")
    print(f"    {TEXT_JSON_FILENAME} — full text corpus + all metrics")


if __name__ == "__main__":
    main()
