#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         LLM OUTPUT TEXT ANALYSIS DASHBOARD                                  ║
║         Bachelor's Thesis Tool — Andrii Pentsak, LNU Lviv, 2026            ║
║                                                                              ║
║  5 Research Modules:                                                         ║
║  1. Zipf's Law — does LLM text follow natural language statistics?           ║
║  2. Model Fingerprint — radar chart of each model's unique style profile     ║
║  3. Quality–Quantity Paradox — TTR degradation curve as text grows           ║
║  4. Prompt Effect vs Model Effect — what drives output volume more?          ║
║  5. Efficiency Map — speed × quality × volume bubble chart                   ║
║                                                                              ║
║  Based on text analysis methods from:                                        ║
║  - Zipf (1935), Heaps (1978) laws                                           ║
║  - Type-Token Ratio, Herdan's C                                              ║
║  - Discourse marker analysis                                                 ║
║  - Lexical density & n-gram statistics                                       ║
║                                                                              ║
║  Usage:                                                                      ║
║    pip install gradio pandas plotly scipy numpy                              ║
║    python dashboard.py                                                       ║
║    Then open http://localhost:7860                                           ║
║                                                                              ║
║  Upload your own text_analysis_results.json to analyze any LLM data!        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import gradio as gr

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "gemma4:e4b":        "#E63946",
    "gemma3:12b":        "#F4A261",
    "qwen3:14b":         "#2A9D8F",
    "granite4.1:8b":     "#457B9D",
    "phi4:14b":          "#9B5DE5",
    "deepseek-r1:1.5b":  "#F15BB5",
    "mistral-nemo:12b":  "#00BBF9",
    "gpt-oss:20b":       "#00F5D4",
    "qwen2.5:3b":        "#FB8500",
}

DEFAULT_COLOR = "#888888"

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", size=12, color="#E0E0E0"),
    xaxis=dict(gridcolor="#2A2A2A", zeroline=False, linecolor="#333"),
    yaxis=dict(gridcolor="#2A2A2A", zeroline=False, linecolor="#333"),
    legend=dict(bgcolor="rgba(20,20,20,0.8)", bordercolor="#333", borderwidth=1),
    colorway=list(MODEL_COLORS.values()),
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(json_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        df = df[df["char_count"] > 100].copy()
        if "config_label" not in df.columns and "config" in df.columns:
            label_map = {"std": "Standard", "creative": "Creative", "precise": "Precise"}
            df["config_label"] = df["config"].map(label_map).fillna(df["config"])
        df["model_short"] = df["model"].apply(lambda x: x.split(":")[0])
        df["color"] = df["model"].map(MODEL_COLORS).fillna(DEFAULT_COLOR)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def get_models(df: pd.DataFrame) -> List[str]:
    return sorted(df["model"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS FUNCTIONS (from course labs)
# ══════════════════════════════════════════════════════════════════════════════

def get_word_frequencies(text: str) -> Counter:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return Counter(words)


def zipf_fit(freq_counter: Counter):
    """
    Zipf's Law: rank × frequency ≈ constant
    f(r) ∝ 1/r^α  (ideal: α ≈ 1)
    Returns ranks, frequencies, fitted line, alpha exponent
    Based on lab 02: Zipf1,2&Pareto laws
    """
    sorted_freqs = sorted(freq_counter.values(), reverse=True)
    ranks = list(range(1, len(sorted_freqs) + 1))
    freqs = sorted_freqs

    # Fit power law in log-log space
    log_ranks = np.log(ranks[:500])
    log_freqs = np.log([max(f, 1) for f in freqs[:500]])
    slope, intercept, r_value, p_value, _ = stats.linregress(log_ranks, log_freqs)
    alpha = -slope
    r2 = r_value ** 2

    fitted = [math.exp(intercept) * (r ** slope) for r in ranks[:500]]
    return ranks[:500], freqs[:500], fitted, round(alpha, 3), round(r2, 4)


def heaps_law_curve(text: str):
    """
    Heaps' Law: V(n) ≈ K × n^β  (vocabulary grows as text grows)
    Ideal β ≈ 0.4–0.6
    Based on lab 04: Heaps law
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 100:
        return [], [], 0, 0

    step = max(1, len(words) // 200)
    n_points = list(range(step, len(words), step))
    vocab_sizes = []
    seen = set()
    word_idx = 0
    for target_n in n_points:
        while word_idx < target_n and word_idx < len(words):
            seen.add(words[word_idx])
            word_idx += 1
        vocab_sizes.append(len(seen))

    if len(n_points) < 10:
        return n_points, vocab_sizes, 0, 0

    log_n = np.log(n_points)
    log_v = np.log(vocab_sizes)
    slope, intercept, r_val, _, _ = stats.linregress(log_n, log_v)
    beta = round(slope, 3)
    K    = round(math.exp(intercept), 3)
    return n_points, vocab_sizes, beta, K


def ngram_stats(text: str, n: int = 2) -> Dict:
    """N-gram frequency analysis — lab 08"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    freq = Counter(ngrams)
    total = sum(freq.values())
    unique = len(freq)
    return {
        "total_ngrams":  total,
        "unique_ngrams": unique,
        "ngram_ttr":     round(unique / total, 4) if total > 0 else 0,
        "top10": [(f"{' '.join(g)}", c) for g, c in freq.most_common(10)],
    }


def sentence_length_distribution(text: str) -> Dict:
    """Sentence length statistics — lab 15"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    if not lengths:
        return {}
    return {
        "lengths": lengths,
        "mean":    round(statistics.mean(lengths), 2),
        "median":  round(statistics.median(lengths), 2),
        "stdev":   round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0,
        "skew":    round(float(stats.skew(lengths)), 3),
        "kurt":    round(float(stats.kurtosis(lengths)), 3),
    }


def repetition_analysis(text: str) -> Dict:
    """Repetition characteristics — lab 16"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    freq = Counter(words)
    total = len(words)
    once = sum(1 for c in freq.values() if c == 1)
    twice = sum(1 for c in freq.values() if c == 2)
    many = sum(1 for c in freq.values() if c > 5)
    return {
        "hapax_ratio":   round(once / len(freq), 3) if freq else 0,
        "dis_ratio":     round(twice / len(freq), 3) if freq else 0,
        "high_freq_pct": round(many / len(freq) * 100, 1) if freq else 0,
        "top_words":     freq.most_common(20),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — ZIPF'S LAW
# ══════════════════════════════════════════════════════════════════════════════

def plot_zipf(df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    fig = go.Figure()

    for model in selected_models:
        recs = df[df["model"] == model]
        if recs.empty:
            continue
        # Use the longest text available for best Zipf fitting
        best_row = recs.loc[recs["char_count"].idxmax()]
        text = best_row.get("full_text", "")
        if not text or len(text) < 500:
            continue

        freq = get_word_frequencies(text)
        ranks, freqs, fitted, alpha, r2 = zipf_fit(freq)
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=ranks, y=freqs,
            mode="markers",
            name=f"{model.split(':')[0]} (α={alpha}, R²={r2})",
            marker=dict(color=color, size=4, opacity=0.6),
            legendgroup=model,
        ))
        fig.add_trace(go.Scatter(
            x=ranks, y=fitted,
            mode="lines",
            name=f"{model.split(':')[0]} fit",
            line=dict(color=color, width=2, dash="dash"),
            legendgroup=model,
            showlegend=False,
        ))

    # Ideal Zipf reference
    if ranks:
        ideal = [freqs[0] / r for r in ranks]
        fig.add_trace(go.Scatter(
            x=ranks, y=ideal,
            mode="lines",
            name="Ideal Zipf (α=1)",
            line=dict(color="#FFFFFF", width=1, dash="dot"),
        ))

    fig.update_layout(
        title="Zipf's Law: Word Rank vs Frequency (log-log)",
        xaxis=dict(title="Rank (log)", type="log", **PLOTLY_TEMPLATE["xaxis"]),
        yaxis=dict(title="Frequency (log)", type="log", **PLOTLY_TEMPLATE["yaxis"]),
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=520,
    )
    return fig


def zipf_table(df: pd.DataFrame, selected_models: List[str]) -> pd.DataFrame:
    rows = []
    for model in selected_models:
        recs = df[df["model"] == model]
        if recs.empty:
            continue
        best_row = recs.loc[recs["char_count"].idxmax()]
        text = best_row.get("full_text", "")
        if not text or len(text) < 500:
            continue
        freq = get_word_frequencies(text)
        _, _, _, alpha, r2 = zipf_fit(freq)
        _, _, beta, K = heaps_law_curve(text)
        rows.append({
            "Model":      model,
            "Zipf α":     alpha,
            "Zipf R²":    r2,
            "Heaps β":    beta,
            "Heaps K":    K,
            "Deviation":  round(abs(alpha - 1.0), 3),
            "Chars":      f"{best_row['char_count']:,}",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MODEL FINGERPRINT
# ══════════════════════════════════════════════════════════════════════════════

FINGERPRINT_PARAMS = [
    ("ttr",                "TTR"),
    ("herdan_c",           "Herdan C"),
    ("avg_sentence_len",   "Avg sent len"),
    ("discourse_per_100w", "Discourse /100w"),
    ("lexical_density",    "Lexical density"),
    ("long_sentences_pct", "Long sent %"),
    ("repeat_word_pct",    "Repeat word %"),
    ("tokens_per_sec",     "Speed (TPS)"),
]


def normalize_param(df: pd.DataFrame, col: str) -> pd.Series:
    mn, mx = df[col].min(), df[col].max()
    if mx == mn:
        return pd.Series([0.5] * len(df), index=df.index)
    return (df[col] - mn) / (mx - mn)


def plot_fingerprint(df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    params = [p for p, _ in FINGERPRINT_PARAMS if p in df.columns]
    labels = [l for p, l in FINGERPRINT_PARAMS if p in df.columns]

    fig = go.Figure()

    for model in selected_models:
        recs = df[df["model"] == model]
        if recs.empty:
            continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        # Average across all configs/prompts
        vals_raw = [recs[p].mean() for p in params]

        # Normalize each param to 0-1 across all models for fair comparison
        vals_norm = []
        for p in params:
            mn, mx = df[p].min(), df[p].max()
            v = recs[p].mean()
            vals_norm.append((v - mn) / (mx - mn) if mx != mn else 0.5)

        fig.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=model.split(":")[0],
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb(", "rgba(") if "rgb" in color else color + "26",
            opacity=0.85,
            hovertemplate="<b>%{theta}</b><br>Normalized: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10, color="#888")),
            angularaxis=dict(tickfont=dict(size=11, color="#CCC")),
            bgcolor="rgba(10,10,20,0.8)",
        ),
        title="Model Fingerprint — Normalized Style Profile",
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=560,
    )
    return fig


def fingerprint_heatmap(df: pd.DataFrame) -> go.Figure:
    params = [p for p, _ in FINGERPRINT_PARAMS if p in df.columns]
    labels = [l for p, l in FINGERPRINT_PARAMS if p in df.columns]
    models = df["model"].unique().tolist()
    model_shorts = [m.split(":")[0] for m in models]

    matrix = []
    for p in params:
        mn, mx = df[p].min(), df[p].max()
        row = []
        for m in models:
            v = df[df["model"] == m][p].mean()
            row.append(round((v - mn) / (mx - mn), 3) if mx != mn else 0.5)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=model_shorts,
        y=labels,
        colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Model: %{x}<br>Param: %{y}<br>Normalized: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Parameter Heatmap — Model vs Feature (normalized 0→1)",
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        xaxis=dict(tickangle=-30, **PLOTLY_TEMPLATE["xaxis"]),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"]),
        height=460,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — QUALITY-QUANTITY PARADOX
# ══════════════════════════════════════════════════════════════════════════════

def plot_ttr_vs_chars(df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    fig = go.Figure()

    # Scatter per model
    for model in selected_models:
        recs = df[df["model"] == model].copy()
        if recs.empty:
            continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        fig.add_trace(go.Scatter(
            x=recs["char_count"],
            y=recs["ttr"],
            mode="markers",
            name=model.split(":")[0],
            marker=dict(color=color, size=10, opacity=0.85,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            text=[f"{r['config_label']} | {r['prompt']}" for _, r in recs.iterrows()],
            hovertemplate="<b>%{fullData.name}</b><br>Chars: %{x:,}<br>TTR: %{y:.3f}<br>%{text}<extra></extra>",
        ))

    # Overall power-law fit
    x_all = df["char_count"].values
    y_all = df["ttr"].values
    valid = (x_all > 0) & (y_all > 0)
    if valid.sum() > 5:
        log_x = np.log(x_all[valid])
        log_y = np.log(y_all[valid])
        slope, intercept, r_val, _, _ = stats.linregress(log_x, log_y)
        x_range = np.linspace(x_all[valid].min(), x_all[valid].max(), 200)
        y_fit = np.exp(intercept) * x_range ** slope
        fig.add_trace(go.Scatter(
            x=x_range, y=y_fit,
            mode="lines",
            name=f"Power-law fit (β={slope:.3f}, R²={r_val**2:.3f})",
            line=dict(color="#FFFFFF", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Quality–Quantity Paradox: TTR vs Character Count",
        xaxis=dict(title="Characters (log)", type="log", **PLOTLY_TEMPLATE["xaxis"]),
        yaxis=dict(title="TTR — Lexical Diversity", **PLOTLY_TEMPLATE["yaxis"]),
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=500,
    )
    return fig


def plot_herdan_comparison(df: pd.DataFrame) -> go.Figure:
    """Herdan's C is length-independent — compare TTR vs Herdan's C"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("TTR (length-dependent)", "Herdan's C (length-independent)"),
        horizontal_spacing=0.12,
    )

    models = df["model"].unique()
    for model in models:
        recs = df[df["model"] == model]
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        short = model.split(":")[0]

        fig.add_trace(go.Box(
            y=recs["ttr"], name=short, marker_color=color,
            showlegend=False, boxmean=True,
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=recs["herdan_c"], name=short, marker_color=color,
            showlegend=False, boxmean=True,
        ), row=1, col=2)

    fig.update_layout(
        title="TTR vs Herdan's C: Which metric is fairer?",
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=460,
        xaxis=dict(tickangle=-35, **PLOTLY_TEMPLATE["xaxis"]),
        xaxis2=dict(tickangle=-35, **PLOTLY_TEMPLATE["xaxis"]),
        yaxis=PLOTLY_TEMPLATE["yaxis"],
        yaxis2=PLOTLY_TEMPLATE["yaxis"],
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — PROMPT EFFECT VS MODEL EFFECT
# ══════════════════════════════════════════════════════════════════════════════

def plot_prompt_vs_model(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg chars by model",
            "Avg chars by prompt type",
            "Avg chars by config",
            "Variance decomposition",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 1. By model
    model_avg = df.groupby("model")["char_count"].mean().sort_values(ascending=False)
    fig.add_trace(go.Bar(
        x=[m.split(":")[0] for m in model_avg.index],
        y=model_avg.values,
        marker_color=[MODEL_COLORS.get(m, DEFAULT_COLOR) for m in model_avg.index],
        showlegend=False,
        hovertemplate="%{x}<br>%{y:,.0f} chars<extra></extra>",
    ), row=1, col=1)

    # 2. By prompt
    prompt_avg = df.groupby("prompt")["char_count"].mean()
    prompt_labels = {
        "prompt_standard":  "Standard",
        "prompt_maxlength": "Maxlength",
        "prompt_detailed":  "Detailed",
    }
    fig.add_trace(go.Bar(
        x=[prompt_labels.get(p, p) for p in prompt_avg.index],
        y=prompt_avg.values,
        marker_color=["#2A9D8F", "#E63946", "#F4A261"],
        showlegend=False,
    ), row=1, col=2)

    # 3. By config
    config_avg = df.groupby("config_label")["char_count"].mean()
    fig.add_trace(go.Bar(
        x=config_avg.index.tolist(),
        y=config_avg.values,
        marker_color=["#457B9D", "#9B5DE5", "#F15BB5"],
        showlegend=False,
    ), row=2, col=1)

    # 4. Variance decomposition (eta-squared proxy)
    from scipy.stats import f_oneway
    groups_model  = [df[df["model"] == m]["char_count"].values for m in df["model"].unique()]
    groups_prompt = [df[df["prompt"] == p]["char_count"].values for p in df["prompt"].unique()]
    groups_config = [df[df["config_label"] == c]["char_count"].values for c in df["config_label"].unique()]

    def eta_sq(groups):
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups if len(g) > 0)
        ss_total   = sum((v - grand_mean)**2 for v in all_vals)
        return round(ss_between / ss_total, 3) if ss_total > 0 else 0

    eta_vals = {
        "Model effect":  eta_sq(groups_model),
        "Prompt effect": eta_sq(groups_prompt),
        "Config effect": eta_sq(groups_config),
    }

    fig.add_trace(go.Bar(
        x=list(eta_vals.keys()),
        y=list(eta_vals.values()),
        marker_color=["#E63946", "#2A9D8F", "#F4A261"],
        text=[f"η²={v}" for v in eta_vals.values()],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        title="What Drives Output Volume? Model vs Prompt vs Config",
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=620,
    )
    for i in range(1, 5):
        suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{suffix}": dict(tickangle=-30, **PLOTLY_TEMPLATE["xaxis"]),
            f"yaxis{suffix}": PLOTLY_TEMPLATE["yaxis"],
        })
    return fig


def prompt_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt in df["prompt"].unique():
        recs = df[df["prompt"] == prompt]
        rows.append({
            "Prompt":           prompt,
            "Avg chars":        f"{recs['char_count'].mean():,.0f}",
            "Avg TTR":          f"{recs['ttr'].mean():.3f}",
            "Avg discourse/100w": f"{recs['discourse_per_100w'].mean():.2f}",
            "Avg sentence len": f"{recs['avg_sentence_len'].mean():.1f}",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — EFFICIENCY MAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_efficiency_map(df: pd.DataFrame, selected_models: List[str],
                        x_metric: str, y_metric: str, size_metric: str) -> go.Figure:
    fig = go.Figure()

    for model in selected_models:
        recs = df[df["model"] == model]
        if recs.empty:
            continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        x_vals = recs[x_metric].values
        y_vals = recs[y_metric].values
        s_vals = recs[size_metric].values
        # Normalize size for bubbles
        s_max = df[size_metric].max()
        sizes = [max(10, min(60, v / s_max * 60)) for v in s_vals]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            name=model.split(":")[0],
            marker=dict(
                color=color,
                size=sizes,
                opacity=0.8,
                line=dict(width=1, color="rgba(255,255,255,0.4)"),
                sizemode="diameter",
            ),
            text=[f"{r['config_label']} | {r['prompt']}<br>{size_metric}: {r[size_metric]:,.0f}"
                  for _, r in recs.iterrows()],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                f"{x_metric}: %{{x:.2f}}<br>"
                f"{y_metric}: %{{y:.2f}}<br>"
                "%{text}<extra></extra>"
            ),
        ))

    METRIC_LABELS = {
        "tokens_per_sec":    "Generation Speed (tokens/sec)",
        "char_count":        "Output Volume (characters)",
        "ttr":               "Lexical Diversity (TTR)",
        "herdan_c":          "Herdan's C",
        "discourse_per_100w":"Discourse Markers / 100 words",
        "avg_sentence_len":  "Avg Sentence Length (words)",
        "lexical_density":   "Lexical Density",
        "long_sentences_pct":"Long Sentences %",
    }

    fig.update_layout(
        title=f"Efficiency Map: {METRIC_LABELS.get(x_metric, x_metric)} vs {METRIC_LABELS.get(y_metric, y_metric)}",
        xaxis=dict(title=METRIC_LABELS.get(x_metric, x_metric), **PLOTLY_TEMPLATE["xaxis"]),
        yaxis=dict(title=METRIC_LABELS.get(y_metric, y_metric), **PLOTLY_TEMPLATE["yaxis"]),
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=540,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — SENTENCE LENGTH DISTRIBUTION (lab 15)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentence_distribution(df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    fig = go.Figure()

    for model in selected_models:
        recs = df[df["model"] == model]
        if recs.empty:
            continue
        best_row = recs.loc[recs["char_count"].idxmax()]
        text = best_row.get("full_text", "")
        if not text or len(text) < 200:
            continue

        sld = sentence_length_distribution(text)
        if not sld or not sld.get("lengths"):
            continue

        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        fig.add_trace(go.Violin(
            y=sld["lengths"],
            name=f"{model.split(':')[0]}<br>μ={sld['mean']} σ={sld['stdev']}",
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color + "40",
        ))

    fig.update_layout(
        title="Sentence Length Distribution per Model (violin plot)",
        yaxis=dict(title="Sentence length (words)", **PLOTLY_TEMPLATE["yaxis"]),
        **{k: v for k, v in PLOTLY_TEMPLATE.items() if k not in ("xaxis", "yaxis", "colorway")},
        height=500,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def summary_cards(df: pd.DataFrame) -> str:
    total_chars  = df["char_count"].sum()
    total_models = df["model"].nunique()
    total_recs   = len(df)
    best_model   = df.groupby("model")["char_count"].mean().idxmax().split(":")[0]
    fastest      = df.groupby("model")["tokens_per_sec"].mean().idxmax().split(":")[0]
    best_ttr     = df.groupby("model")["herdan_c"].mean().idxmax().split(":")[0]
    stop_pct     = round(sum(df["done_reason"] == "stop") / len(df) * 100)

    return f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
  <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;text-align:center">
    <div style="font-size:11px;color:#888;margin-bottom:4px">TOTAL CHARACTERS</div>
    <div style="font-size:22px;font-weight:600;color:#2A9D8F">{total_chars:,}</div>
  </div>
  <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;text-align:center">
    <div style="font-size:11px;color:#888;margin-bottom:4px">MODELS COMPARED</div>
    <div style="font-size:22px;font-weight:600;color:#E63946">{total_models}</div>
    <div style="font-size:11px;color:#555">{total_recs} records total</div>
  </div>
  <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;text-align:center">
    <div style="font-size:11px;color:#888;margin-bottom:4px">FASTEST MODEL</div>
    <div style="font-size:18px;font-weight:600;color:#F4A261">{fastest}</div>
    <div style="font-size:11px;color:#555">{round(df.groupby('model')['tokens_per_sec'].mean().max())} t/s</div>
  </div>
  <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:16px;text-align:center">
    <div style="font-size:11px;color:#888;margin-bottom:4px">NATURAL STOP %</div>
    <div style="font-size:22px;font-weight:600;color:#9B5DE5">{stop_pct}%</div>
    <div style="font-size:11px;color:#555">models finish naturally</div>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
body { background: #0d0d1a !important; }
.gradio-container { background: #0d0d1a !important; font-family: 'JetBrains Mono', monospace; }
.tab-nav button { background: #1a1a2e !important; color: #888 !important; border: 1px solid #333 !important; }
.tab-nav button.selected { background: #2A9D8F !important; color: #fff !important; }
h1, h2, h3 { color: #E0E0E0 !important; }
.svelte-1gfkn6j { background: #1a1a2e !important; }
footer { display: none !important; }
.prose { color: #CCC !important; }
"""

METRICS_CHOICES = [
    ("Generation Speed (TPS)",      "tokens_per_sec"),
    ("Output Volume (chars)",        "char_count"),
    ("Lexical Diversity (TTR)",      "ttr"),
    ("Herdan's C",                   "herdan_c"),
    ("Discourse Markers /100w",      "discourse_per_100w"),
    ("Avg Sentence Length",          "avg_sentence_len"),
    ("Lexical Density",              "lexical_density"),
    ("Long Sentences %",             "long_sentences_pct"),
]


def build_ui():
    # Try to load default data
    default_json = "text_analysis_results.json"
    df_default = load_data(default_json) if Path(default_json).exists() else None

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="LLM Text Analysis Dashboard",
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    ) as demo:

        gr.HTML("""
        <div style="padding:20px 0;border-bottom:1px solid #222;margin-bottom:20px">
          <h1 style="margin:0;font-size:24px;color:#2A9D8F;letter-spacing:2px">
            ⬡ LLM OUTPUT TEXT ANALYSIS DASHBOARD
          </h1>
          <p style="margin:6px 0 0;color:#666;font-size:13px">
            Bachelor's Thesis Tool — Andrii Pentsak, LNU Lviv, 2026 |
            Upload your own JSON to analyze any LLM collection
          </p>
        </div>
        """)

        # ── Data upload ───────────────────────────────────────────────────────
        with gr.Row():
            json_upload = gr.File(
                label="Upload text_analysis_results.json",
                file_types=[".json"],
                value=default_json if df_default is not None else None,
            )
            load_btn = gr.Button("Load Data", variant="primary", scale=0)

        status_html = gr.HTML(
            value=summary_cards(df_default) if df_default is not None else
                  "<p style='color:#666'>Upload a JSON file to begin</p>"
        )

        df_state = gr.State(value=df_default)

        def on_load(file, current_df):
            if file is None:
                return current_df, "<p style='color:#E63946'>No file selected</p>"
            df = load_data(file.name)
            if df is None:
                return current_df, "<p style='color:#E63946'>Failed to load file</p>"
            return df, summary_cards(df)

        load_btn.click(on_load, [json_upload, df_state], [df_state, status_html])

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Zipf's Law ─────────────────────────────────────────────
            with gr.Tab("① Zipf's Law"):
                gr.Markdown("""
**Zipf's Law** (1935): In natural text, word frequency is inversely proportional to rank.
`f(r) ∝ 1/r^α` — ideal human text has **α ≈ 1.0**. LLM-generated text may deviate.
**Heaps' Law** measures vocabulary growth: `V(n) ≈ K·n^β`, ideal **β ≈ 0.5**.
""")
                with gr.Row():
                    zipf_model_select = gr.CheckboxGroup(
                        label="Select models",
                        choices=get_models(df_default) if df_default is not None else [],
                        value=get_models(df_default)[:4] if df_default is not None else [],
                    )

                zipf_plot = gr.Plot()
                zipf_table_out = gr.Dataframe(label="Zipf & Heaps statistics per model")
                run_zipf_btn = gr.Button("Run Zipf Analysis", variant="primary")

                def run_zipf(models, df):
                    if df is None or not models:
                        return None, pd.DataFrame()
                    fig = plot_zipf(df, models)
                    tbl = zipf_table(df, models)
                    return fig, tbl

                run_zipf_btn.click(run_zipf, [zipf_model_select, df_state],
                                   [zipf_plot, zipf_table_out])
                df_state.change(
                    lambda df: (gr.CheckboxGroup(choices=get_models(df) if df is not None else [],
                                                 value=get_models(df)[:4] if df is not None else [])),
                    [df_state], [zipf_model_select]
                )

            # ── Tab 2: Model Fingerprint ──────────────────────────────────────
            with gr.Tab("② Model Fingerprint"):
                gr.Markdown("""
**Model Fingerprint**: Each LLM has a unique stylistic profile across 8 dimensions.
Radar chart shows normalized values — use it to visually distinguish models at a glance.
The heatmap reveals which models are most similar and where they diverge.
""")
                with gr.Row():
                    fp_model_select = gr.CheckboxGroup(
                        label="Select models",
                        choices=get_models(df_default) if df_default is not None else [],
                        value=get_models(df_default) if df_default is not None else [],
                    )

                fp_radar = gr.Plot(label="Radar — Style Profile")
                fp_heat  = gr.Plot(label="Heatmap — Model vs Parameter")
                run_fp_btn = gr.Button("Generate Fingerprint", variant="primary")

                def run_fp(models, df):
                    if df is None:
                        return None, None
                    return plot_fingerprint(df, models or get_models(df)), fingerprint_heatmap(df)

                run_fp_btn.click(run_fp, [fp_model_select, df_state], [fp_radar, fp_heat])

            # ── Tab 3: Quality-Quantity Paradox ───────────────────────────────
            with gr.Tab("③ Quality–Quantity Paradox"):
                gr.Markdown("""
**The Paradox**: More text → less diverse vocabulary (lower TTR).
But is this a universal law, or model-specific?
**Herdan's C** is a length-independent alternative to TTR — does it tell a different story?
Correlation coefficient shown: how strong is the TTR–length relationship per model?
""")
                with gr.Row():
                    qq_model_select = gr.CheckboxGroup(
                        label="Select models",
                        choices=get_models(df_default) if df_default is not None else [],
                        value=get_models(df_default) if df_default is not None else [],
                    )

                qq_scatter = gr.Plot(label="TTR vs Character Count")
                qq_box     = gr.Plot(label="TTR vs Herdan's C comparison")
                run_qq_btn = gr.Button("Analyze Paradox", variant="primary")

                def run_qq(models, df):
                    if df is None:
                        return None, None
                    return (plot_ttr_vs_chars(df, models or get_models(df)),
                            plot_herdan_comparison(df))

                run_qq_btn.click(run_qq, [qq_model_select, df_state], [qq_scatter, qq_box])

            # ── Tab 4: Prompt vs Model Effect ─────────────────────────────────
            with gr.Tab("④ Prompt vs Model Effect"):
                gr.Markdown("""
**Key question**: What drives output volume more — which MODEL you choose, or HOW you prompt it?
**η² (eta-squared)** measures the proportion of variance explained by each factor (0→1).
Higher η² = stronger effect on output volume.
""")
                pme_plot   = gr.Plot(label="Variance Decomposition")
                pme_table  = gr.Dataframe(label="Statistics by Prompt Type")
                run_pme_btn = gr.Button("Decompose Effects", variant="primary")

                def run_pme(df):
                    if df is None:
                        return None, pd.DataFrame()
                    return plot_prompt_vs_model(df), prompt_effect_table(df)

                run_pme_btn.click(run_pme, [df_state], [pme_plot, pme_table])

            # ── Tab 5: Efficiency Map ─────────────────────────────────────────
            with gr.Tab("⑤ Efficiency Map"):
                gr.Markdown("""
**Efficiency Map**: Interactive bubble chart — choose any 3 metrics for X, Y, and bubble size.
Find the optimal model for your use case: fastest? most diverse? highest volume?
""")
                with gr.Row():
                    eff_x = gr.Dropdown(
                        label="X axis",
                        choices=[(l, v) for l, v in METRICS_CHOICES],
                        value="tokens_per_sec",
                    )
                    eff_y = gr.Dropdown(
                        label="Y axis",
                        choices=[(l, v) for l, v in METRICS_CHOICES],
                        value="char_count",
                    )
                    eff_sz = gr.Dropdown(
                        label="Bubble size",
                        choices=[(l, v) for l, v in METRICS_CHOICES],
                        value="ttr",
                    )

                eff_model_select = gr.CheckboxGroup(
                    label="Select models",
                    choices=get_models(df_default) if df_default is not None else [],
                    value=get_models(df_default) if df_default is not None else [],
                )
                eff_plot = gr.Plot()
                run_eff_btn = gr.Button("Generate Map", variant="primary")

                def run_eff(x, y, sz, models, df):
                    if df is None:
                        return None
                    return plot_efficiency_map(df, models or get_models(df), x, y, sz)

                run_eff_btn.click(run_eff, [eff_x, eff_y, eff_sz, eff_model_select, df_state],
                                  [eff_plot])

            # ── Tab 6: Sentence Distribution (lab 15) ────────────────────────
            with gr.Tab("⑥ Sentence Distribution"):
                gr.Markdown("""
**Sentence Length Distribution** (Lab 15 — word/sentence length analysis).
Violin plot shows distribution shape per model: mean, median, spread, skewness.
Long-tailed distribution → complex academic style. Short sentences → conversational style.
""")
                with gr.Row():
                    sd_model_select = gr.CheckboxGroup(
                        label="Select models",
                        choices=get_models(df_default) if df_default is not None else [],
                        value=get_models(df_default)[:5] if df_default is not None else [],
                    )

                sd_plot = gr.Plot()
                run_sd_btn = gr.Button("Plot Distributions", variant="primary")

                def run_sd(models, df):
                    if df is None:
                        return None
                    return plot_sentence_distribution(df, models or get_models(df))

                run_sd_btn.click(run_sd, [sd_model_select, df_state], [sd_plot])

            # ── Tab 7: Export ─────────────────────────────────────────────────
            with gr.Tab("⑦ Export & Report"):
                gr.Markdown("Export processed statistics for your thesis.")
                export_btn = gr.Button("Generate CSV Summary", variant="primary")
                export_file = gr.File(label="Download CSV")

                def export_csv(df):
                    if df is None:
                        return None
                    out_path = "/tmp/llm_analysis_export.csv"
                    cols = [c for c in df.columns if c not in ("full_text", "txt_file", "color")]
                    df[cols].to_csv(out_path, index=False)
                    return out_path

                export_btn.click(export_csv, [df_state], [export_file])

                gr.Markdown("""
---
**About this tool**

Built for the bachelor's thesis *"Analysis of Output Text Parameters in Modern Generative Language Models"*
by Andrii Pentsak, Ivan Franko National University of Lviv, 2026.

Methods used:
- **Zipf's Law** (1935) — rank-frequency distribution
- **Heaps' Law** (1978) — vocabulary growth
- **TTR & Herdan's C** — lexical diversity (length-independent)
- **η² variance decomposition** — effect size analysis
- **Violin plots** — sentence length distributions (Lab 15)
- **Radar fingerprinting** — multi-dimensional style profiling

To use with your own data:
1. Run `text_analysis_v3.py` on your Ollama instance
2. Upload the resulting `text_analysis_results.json` here
3. All analyses run automatically on your data
""")

        # ── Auto-run on load ──────────────────────────────────────────────────
        if df_default is not None:
            demo.load(
                lambda df: (
                    plot_zipf(df, get_models(df)[:4]) if df is not None else None,
                    plot_fingerprint(df, get_models(df)) if df is not None else None,
                    fingerprint_heatmap(df) if df is not None else None,
                    plot_ttr_vs_chars(df, get_models(df)) if df is not None else None,
                    plot_herdan_comparison(df) if df is not None else None,
                    plot_prompt_vs_model(df) if df is not None else None,
                    prompt_effect_table(df) if df is not None else pd.DataFrame(),
                    plot_efficiency_map(df, get_models(df), "tokens_per_sec", "char_count", "ttr") if df is not None else None,
                    plot_sentence_distribution(df, get_models(df)[:5]) if df is not None else None,
                ),
                inputs=[df_state],
                outputs=[zipf_plot, fp_radar, fp_heat, qq_scatter, qq_box,
                         pme_plot, pme_table, eff_plot, sd_plot],
            )

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Text Analysis Dashboard")
    parser.add_argument("--data",   default="text_analysis_results.json",
                        help="Path to JSON results file")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Text Analysis Dashboard")
    print(f"  Data: {args.data}")
    print(f"  URL:  http://localhost:{args.port}")
    print("=" * 60)

    if not Path(args.data).exists():
        print(f"\n[!] {args.data} not found.")
        print("    Run text_analysis_v3.py first, or upload JSON in the UI.\n")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
