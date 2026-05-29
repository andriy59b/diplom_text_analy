#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  АНАЛІЗ ПАРАМЕТРІВ ВИХІДНИХ ТЕКСТІВ LLM — ДАШБОРД                          ║
║  Бакалаврська робота — Пенцак Андрій, ЛНУ ім. Франка, 2026                 ║
║                                                                              ║
║  Запуск:                                                                     ║
║    pip install gradio pandas plotly scipy numpy                              ║
║    python dashboard.py                                                       ║
║    Відкрий http://localhost:7860                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json, math, re, statistics, zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import gradio as gr

# ══════════════════════════════════════════════════════════════════════════════
# КОЛЬОРИ МОДЕЛЕЙ
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

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

AXIS_STYLE = dict(gridcolor="#2A2A2A", zeroline=False, linecolor="#444")
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,25,0.9)",
    font=dict(family="monospace", size=12, color="#D0D0D0"),
    legend=dict(bgcolor="rgba(20,20,30,0.85)", bordercolor="#444", borderwidth=1),
)

# ══════════════════════════════════════════════════════════════════════════════
# ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════

# Global text cache: record_id -> text
_text_cache: Dict[str, str] = {}
_zip_archive: Optional[zipfile.ZipFile] = None

def load_texts_from_zip(zip_path: str) -> int:
    global _zip_archive, _text_cache
    try:
        _zip_archive = zipfile.ZipFile(zip_path, 'r')
        _text_cache = {}
        for name in _zip_archive.namelist():
            if name.endswith('.txt'):
                stem = Path(name).stem
                try:
                    text = _zip_archive.read(name).decode('utf-8')
                    _text_cache[stem] = text
                except Exception:
                    pass
        return len(_text_cache)
    except Exception as e:
        print(f"Zip error: {e}")
        return 0

def get_text(row: pd.Series) -> str:
    if 'full_text' in row and row['full_text']:
        return str(row['full_text'])
    record_id = str(row.get('record_id', ''))
    safe_id = record_id.replace(':', '-').replace('/', '-')
    if safe_id in _text_cache:
        return _text_cache[safe_id]
    txt_file = str(row.get('txt_file', ''))
    stem = Path(txt_file).stem
    if stem in _text_cache:
        return _text_cache[stem]
    return ""

def load_json(json_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(json_path, encoding='utf-8') as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        df = df[df['char_count'] > 100].copy()
        if 'config_label' not in df.columns and 'config' in df.columns:
            label_map = {'std':'Стандартний','creative':'Творчий','precise':'Точний'}
            df['config_label'] = df['config'].map(label_map).fillna(df['config'])
        df['model_short'] = df['model'].apply(lambda x: x.split(':')[0])
        df['color'] = df['model'].map(MODEL_COLORS).fillna(DEFAULT_COLOR)
        return df
    except Exception as e:
        print(f"JSON error: {e}")
        return None

def get_models(df: pd.DataFrame) -> List[str]:
    return sorted(df['model'].unique().tolist())

# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦІЇ АНАЛІЗУ ТЕКСТУ (за методами курсу)
# ══════════════════════════════════════════════════════════════════════════════

def word_freq(text: str) -> Counter:
    return Counter(re.findall(r'\b[a-zA-Z]+\b', text.lower()))

def zipf_fit(freq_counter: Counter) -> Tuple:
    """Закон Зіпфа (лаб. 02): f(r) ∝ 1/r^α, ідеал α≈1.0"""
    sorted_f = sorted(freq_counter.values(), reverse=True)
    ranks = list(range(1, min(501, len(sorted_f)+1)))
    freqs = sorted_f[:500]
    if len(ranks) < 10:
        return ranks, freqs, [], 0.0, 0.0
    log_r = np.log(ranks)
    log_f = np.log([max(v,1) for v in freqs])
    slope, intercept, r_val, _, _ = stats.linregress(log_r, log_f)
    alpha = -slope
    r2 = r_val**2
    fitted = [math.exp(intercept) * r**slope for r in ranks]
    return ranks, freqs, fitted, round(alpha,3), round(r2,4)

def heaps_curve(text: str) -> Tuple:
    """Закон Хіпса (лаб. 04): V(n) ≈ K·n^β, ідеал β≈0.5"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 100:
        return [], [], 0.0, 0.0
    step = max(1, len(words)//150)
    n_pts, vocab = [], []
    seen = set()
    for i, w in enumerate(words):
        seen.add(w)
        if i % step == 0:
            n_pts.append(i+1)
            vocab.append(len(seen))
    if len(n_pts) < 5:
        return n_pts, vocab, 0.0, 0.0
    slope, intercept, r_val, _, _ = stats.linregress(np.log(n_pts), np.log(vocab))
    return n_pts, vocab, round(slope,3), round(math.exp(intercept),3)

def sent_lengths(text: str) -> List[int]:
    """Довжини речень (лаб. 15)"""
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    return [len(re.findall(r'\b\w+\b', s)) for s in sents]

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 1 — ЗАКОН ЗІПФА
# ══════════════════════════════════════════════════════════════════════════════

def plot_zipf(df: pd.DataFrame, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    last_ranks, last_freqs = [], []

    for model in selected:
        recs = df[df['model'] == model]
        if recs.empty: continue
        best = recs.loc[recs['char_count'].idxmax()]
        text = get_text(best)
        if len(text) < 500: continue

        freq = word_freq(text)
        ranks, freqs, fitted, alpha, r2 = zipf_fit(freq)
        if not ranks: continue
        last_ranks, last_freqs = ranks, freqs
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=ranks, y=freqs, mode='markers',
            name=f"{model.split(':')[0]} (α={alpha}, R²={r2})",
            marker=dict(color=color, size=4, opacity=0.55),
        ))
        fig.add_trace(go.Scatter(
            x=ranks, y=fitted, mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False,
        ))

    if last_ranks and last_freqs:
        ideal = [last_freqs[0]/r for r in last_ranks]
        fig.add_trace(go.Scatter(
            x=last_ranks, y=ideal, mode='lines',
            name='Ідеал Зіпфа (α=1)',
            line=dict(color='#FFFFFF', width=1, dash='dot'),
        ))

    fig.update_layout(
        title='Закон Зіпфа: Ранг слова vs Частота (log-log)',
        xaxis=dict(title='Ранг (log)', type='log', **AXIS_STYLE),
        yaxis=dict(title='Частота (log)', type='log', **AXIS_STYLE),
        height=500, **LAYOUT_BASE,
    )
    return fig

def zipf_stats_table(df: pd.DataFrame, selected: List[str]) -> pd.DataFrame:
    rows = []
    for model in selected:
        recs = df[df['model'] == model]
        if recs.empty: continue
        best = recs.loc[recs['char_count'].idxmax()]
        text = get_text(best)
        if len(text) < 500: continue
        freq = word_freq(text)
        _, _, _, alpha, r2 = zipf_fit(freq)
        _, _, beta, K = heaps_curve(text)
        rows.append({
            'Модель': model,
            'Показник Зіпфа α': alpha,
            'R² (якість апрокс.)': r2,
            'Відхилення від 1.0': round(abs(alpha-1.0),3),
            'Показник Хіпса β': beta,
            'K (Хіпс)': K,
            'Кількість символів': f"{best['char_count']:,}",
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 2 — ВІДБИТОК МОДЕЛІ
# ══════════════════════════════════════════════════════════════════════════════

FP_PARAMS = [
    ('ttr',                'TTR'),
    ('herdan_c',           'Херданс C'),
    ('avg_sentence_len',   'Довж. речення'),
    ('discourse_per_100w', 'Дискурс/100сл'),
    ('lexical_density',    'Лекс. густота'),
    ('long_sentences_pct', 'Довгі речення %'),
    ('repeat_word_pct',    'Повтори слів %'),
    ('tokens_per_sec',     'Швидкість (TPS)'),
]

def plot_fingerprint(df: pd.DataFrame, selected: List[str]) -> go.Figure:
    params = [p for p,_ in FP_PARAMS if p in df.columns]
    labels = [l for p,l in FP_PARAMS if p in df.columns]
    fig = go.Figure()

    for model in selected:
        recs = df[df['model'] == model]
        if recs.empty: continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        norm = []
        for p in params:
            mn, mx = df[p].min(), df[p].max()
            v = recs[p].mean()
            norm.append((v-mn)/(mx-mn) if mx!=mn else 0.5)

        fig.add_trace(go.Scatterpolar(
            r=norm+[norm[0]], theta=labels+[labels[0]],
            fill='toself',
            name=model.split(':')[0],
            line=dict(color=color, width=2),
            fillcolor=hex_to_rgba(color, 0.15),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9,color='#777')),
            angularaxis=dict(tickfont=dict(size=11,color='#CCC')),
            bgcolor='rgba(10,10,20,0.85)',
        ),
        title='Відбиток моделі — Нормалізований стильовий профіль',
        height=560, **LAYOUT_BASE,
    )
    return fig

def plot_heatmap(df: pd.DataFrame) -> go.Figure:
    params = [p for p,_ in FP_PARAMS if p in df.columns]
    labels = [l for p,l in FP_PARAMS if p in df.columns]
    models = df['model'].unique().tolist()

    matrix = []
    for p in params:
        mn, mx = df[p].min(), df[p].max()
        row = []
        for m in models:
            v = df[df['model']==m][p].mean()
            row.append(round((v-mn)/(mx-mn),3) if mx!=mn else 0.5)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=[m.split(':')[0] for m in models], y=labels,
        colorscale='Viridis',
        text=[[f'{v:.2f}' for v in row] for row in matrix],
        texttemplate='%{text}', textfont=dict(size=10),
    ))
    fig.update_layout(
        title='Теплова карта параметрів (Модель × Характеристика)',
        xaxis=dict(tickangle=-30, **AXIS_STYLE),
        yaxis=AXIS_STYLE,
        height=460, **LAYOUT_BASE,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 3 — ПАРАДОКС ЯКІСТЬ–КІЛЬКІСТЬ
# ══════════════════════════════════════════════════════════════════════════════

def plot_ttr_scatter(df: pd.DataFrame, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for model in selected:
        recs = df[df['model']==model]
        if recs.empty: continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        fig.add_trace(go.Scatter(
            x=recs['char_count'], y=recs['ttr'], mode='markers',
            name=model.split(':')[0],
            marker=dict(color=color, size=11, opacity=0.85,
                        line=dict(width=1, color='rgba(255,255,255,0.3)')),
            text=[f"{r['config_label']} | {r['prompt']}" for _,r in recs.iterrows()],
            hovertemplate='<b>%{fullData.name}</b><br>Символів: %{x:,}<br>TTR: %{y:.3f}<br>%{text}<extra></extra>',
        ))

    x_all = df['char_count'].values
    y_all = df['ttr'].values
    valid = (x_all>0)&(y_all>0)
    if valid.sum() > 5:
        slope, intercept, r_val, _, _ = stats.linregress(np.log(x_all[valid]), np.log(y_all[valid]))
        xr = np.linspace(x_all[valid].min(), x_all[valid].max(), 200)
        fig.add_trace(go.Scatter(
            x=xr, y=np.exp(intercept)*xr**slope, mode='lines',
            name=f'Степ. апрокс. (β={slope:.3f}, R²={r_val**2:.3f})',
            line=dict(color='#FFFFFF', width=2, dash='dash'),
        ))

    fig.update_layout(
        title='Парадокс якість–кількість: TTR vs Кількість символів',
        xaxis=dict(title='Символів (log)', type='log', **AXIS_STYLE),
        yaxis=dict(title='TTR — лексична різноманітність', **AXIS_STYLE),
        height=500, **LAYOUT_BASE,
    )
    return fig

def plot_herdan_box(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(1,2,
        subplot_titles=('TTR (залежить від довжини)','Херданс C (не залежить від довжини)'),
        horizontal_spacing=0.12)

    for model in df['model'].unique():
        recs = df[df['model']==model]
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        short = model.split(':')[0]
        fig.add_trace(go.Box(y=recs['ttr'], name=short, marker_color=color,
                             showlegend=False, boxmean=True), row=1, col=1)
        fig.add_trace(go.Box(y=recs['herdan_c'], name=short, marker_color=color,
                             showlegend=False, boxmean=True), row=1, col=2)

    fig.update_layout(title='TTR vs Херданс C: яка метрика справедливіша?',
                      height=460, **LAYOUT_BASE,
                      xaxis=dict(tickangle=-35,**AXIS_STYLE), xaxis2=dict(tickangle=-35,**AXIS_STYLE),
                      yaxis=AXIS_STYLE, yaxis2=AXIS_STYLE)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 4 — ЕФЕКТ ПРОМТУ VS ЕФЕКТ МОДЕЛІ
# ══════════════════════════════════════════════════════════════════════════════

def eta_squared(groups: List[np.ndarray]) -> float:
    groups = [g for g in groups if len(g)>0]
    if not groups: return 0.0
    all_v = np.concatenate(groups)
    gm = np.mean(all_v)
    ssb = sum(len(g)*(np.mean(g)-gm)**2 for g in groups)
    sst = sum((v-gm)**2 for v in all_v)
    return round(ssb/sst, 3) if sst>0 else 0.0

def plot_effects(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(2,2,
        subplot_titles=('Середня кількість символів за моделлю',
                        'Середня кількість символів за типом промту',
                        'Середня кількість символів за конфігурацією',
                        'Розклад дисперсії (η²)'),
        vertical_spacing=0.2, horizontal_spacing=0.12)

    ma = df.groupby('model')['char_count'].mean().sort_values(ascending=False)
    fig.add_trace(go.Bar(
        x=[m.split(':')[0] for m in ma.index], y=ma.values,
        marker_color=[MODEL_COLORS.get(m,DEFAULT_COLOR) for m in ma.index],
        showlegend=False), row=1, col=1)

    prompt_labels = {'prompt_standard':'Стандартний','prompt_maxlength':'Макс. довжина','prompt_detailed':'Детальний'}
    pa = df.groupby('prompt')['char_count'].mean()
    fig.add_trace(go.Bar(
        x=[prompt_labels.get(p,p) for p in pa.index], y=pa.values,
        marker_color=['#2A9D8F','#E63946','#F4A261'], showlegend=False), row=1, col=2)

    ca = df.groupby('config_label')['char_count'].mean()
    fig.add_trace(go.Bar(
        x=ca.index.tolist(), y=ca.values,
        marker_color=['#457B9D','#9B5DE5','#F15BB5'], showlegend=False), row=2, col=1)

    gm = [df[df['model']==m]['char_count'].values for m in df['model'].unique()]
    gp = [df[df['prompt']==p]['char_count'].values for p in df['prompt'].unique()]
    gc = [df[df['config_label']==c]['char_count'].values for c in df['config_label'].unique()]
    eta = {'Ефект моделі': eta_squared(gm),
           'Ефект промту': eta_squared(gp),
           'Ефект конфігу': eta_squared(gc)}
    fig.add_trace(go.Bar(
        x=list(eta.keys()), y=list(eta.values()),
        marker_color=['#E63946','#2A9D8F','#F4A261'],
        text=[f"η²={v}" for v in eta.values()], textposition='outside',
        showlegend=False), row=2, col=2)

    fig.update_layout(title='Що визначає обсяг тексту? Модель vs Промт vs Конфігурація',
                      height=640, **LAYOUT_BASE)
    for i in range(1,5):
        s = '' if i==1 else str(i)
        fig.update_layout(**{f'xaxis{s}': dict(tickangle=-30,**AXIS_STYLE),
                              f'yaxis{s}': AXIS_STYLE})
    return fig

def prompt_table(df: pd.DataFrame) -> pd.DataFrame:
    prompt_labels = {'prompt_standard':'Стандартний','prompt_maxlength':'Макс. довжина','prompt_detailed':'Детальний'}
    rows = []
    for p in df['prompt'].unique():
        recs = df[df['prompt']==p]
        rows.append({
            'Промт': prompt_labels.get(p,p),
            'Сер. символів': f"{recs['char_count'].mean():,.0f}",
            'Сер. TTR': f"{recs['ttr'].mean():.3f}",
            'Сер. дискурс/100сл': f"{recs['discourse_per_100w'].mean():.2f}",
            'Сер. довж. речення': f"{recs['avg_sentence_len'].mean():.1f}",
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 5 — КАРТА ЕФЕКТИВНОСТІ
# ══════════════════════════════════════════════════════════════════════════════

METRIC_UA = {
    'tokens_per_sec':    'Швидкість генерації (токенів/сек)',
    'char_count':        "Обсяг тексту (символів)",
    'ttr':               'Лексична різноманітність (TTR)',
    'herdan_c':          'Херданс C',
    'discourse_per_100w':'Дискурсивні маркери / 100 слів',
    'avg_sentence_len':  'Сер. довжина речення (слів)',
    'lexical_density':   'Лексична густота',
    'long_sentences_pct':'Довгі речення %',
}

def plot_efficiency(df: pd.DataFrame, selected: List[str], x_m: str, y_m: str, sz_m: str) -> go.Figure:
    fig = go.Figure()
    sz_max = df[sz_m].max()
    for model in selected:
        recs = df[df['model']==model]
        if recs.empty: continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        sizes = [max(10, min(60, v/sz_max*60)) for v in recs[sz_m].values]
        fig.add_trace(go.Scatter(
            x=recs[x_m], y=recs[y_m], mode='markers',
            name=model.split(':')[0],
            marker=dict(color=color, size=sizes, opacity=0.82,
                        line=dict(width=1, color='rgba(255,255,255,0.3)'),
                        sizemode='diameter'),
            text=[f"{r['config_label']} | {r['prompt']}<br>{sz_m}: {r[sz_m]:,.1f}"
                  for _,r in recs.iterrows()],
            hovertemplate='<b>%{fullData.name}</b><br>%{x:.2f} / %{y:.2f}<br>%{text}<extra></extra>',
        ))
    fig.update_layout(
        title=f"Карта ефективності: {METRIC_UA.get(x_m,x_m)} vs {METRIC_UA.get(y_m,y_m)}",
        xaxis=dict(title=METRIC_UA.get(x_m,x_m), **AXIS_STYLE),
        yaxis=dict(title=METRIC_UA.get(y_m,y_m), **AXIS_STYLE),
        height=540, **LAYOUT_BASE,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 6 — РОЗПОДІЛ ДОВЖИН РЕЧЕНЬ (лаб. 15)
# ══════════════════════════════════════════════════════════════════════════════

def plot_violin(df: pd.DataFrame, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for model in selected:
        recs = df[df['model']==model]
        if recs.empty: continue
        best = recs.loc[recs['char_count'].idxmax()]
        text = get_text(best)
        if len(text) < 200: continue
        lengths = sent_lengths(text)
        if not lengths: continue
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        mu = round(statistics.mean(lengths),1)
        sd = round(statistics.stdev(lengths),1) if len(lengths)>1 else 0
        fig.add_trace(go.Violin(
            y=lengths,
            name=f"{model.split(':')[0]}<br>μ={mu} σ={sd}",
            box_visible=True, meanline_visible=True,
            line_color=color,
            fillcolor=hex_to_rgba(color, 0.25),
        ))
    fig.update_layout(
        title='Розподіл довжин речень за моделями (скрипковий графік)',
        yaxis=dict(title='Довжина речення (слів)', **AXIS_STYLE),
        height=500, **LAYOUT_BASE,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CARDS
# ══════════════════════════════════════════════════════════════════════════════

def summary_html(df: pd.DataFrame) -> str:
    total_chars  = df['char_count'].sum()
    n_models     = df['model'].nunique()
    n_recs       = len(df)
    fastest      = df.groupby('model')['tokens_per_sec'].mean().idxmax().split(':')[0]
    max_tps      = round(df.groupby('model')['tokens_per_sec'].mean().max())
    best_vol     = df.groupby('model')['char_count'].mean().idxmax().split(':')[0]
    stop_pct     = round(sum(df['done_reason']=='stop')/len(df)*100)

    def card(label, value, sub=''):
        return f"""<div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;
padding:14px 12px;text-align:center;min-width:140px">
<div style="font-size:10px;color:#666;letter-spacing:1px;margin-bottom:4px">{label}</div>
<div style="font-size:20px;font-weight:600;color:#2A9D8F">{value}</div>
{f'<div style="font-size:10px;color:#555;margin-top:2px">{sub}</div>' if sub else ''}
</div>"""

    cards = (
        card('ВСЬОГО СИМВОЛІВ', f'{total_chars:,}') +
        card('МОДЕЛЕЙ', str(n_models), f'{n_recs} записів') +
        card('НАЙШВИДША', fastest, f'{max_tps} т/сек') +
        card('НАЙБІЛЬШИЙ ОБСЯГ', best_vol) +
        card('ПРИРОДНИЙ СТОП %', f'{stop_pct}%', 'моделі завершують самі')
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">{cards}</div>'

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

METRICS_UA = [(v,k) for k,v in METRIC_UA.items()]

# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 7 — ПОРІВНЯННЯ КОНФІГУРАЦІЙ (Standard / Creative / Precise)
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_UA  = {'Standard': 'Стандартний', 'Creative': 'Творчий', 'Precise': 'Точний'}
CONFIG_COLORS = {'Standard': '#457B9D', 'Creative': '#E63946', 'Precise': '#2A9D8F'}
PROMPT_UA  = {
    'prompt_standard':  'Стандартний промт',
    'prompt_maxlength': 'Макс. довжина',
    'prompt_detailed':  'Детальний промт',
}
PROMPT_COLORS = {
    'prompt_standard':  '#2A9D8F',
    'prompt_maxlength': '#E63946',
    'prompt_detailed':  '#F4A261',
}

def plot_config_comparison(df: pd.DataFrame, metric: str, selected_models: List[str]) -> go.Figure:
    """Grouped bar: для кожної моделі — 3 конфіги поруч"""
    models = [m for m in selected_models if m in df['model'].values]
    fig = go.Figure()

    for cfg, color in CONFIG_COLORS.items():
        vals = []
        for m in models:
            recs = df[(df['model']==m) & (df['config_label']==cfg)]
            vals.append(recs[metric].mean() if not recs.empty else 0)
        fig.add_trace(go.Bar(
            name=CONFIG_UA.get(cfg, cfg),
            x=[m.split(':')[0] for m in models],
            y=vals,
            marker_color=color,
            text=[f"{v:,.1f}" for v in vals],
            textposition='outside',
            textfont=dict(size=9),
        ))

    fig.update_layout(
        barmode='group',
        title=f'{METRIC_UA.get(metric, metric)} — порівняння конфігурацій по моделях',
        xaxis=dict(title='Модель', tickangle=-30, **AXIS_STYLE),
        yaxis=dict(title=METRIC_UA.get(metric, metric), **AXIS_STYLE),
        height=480, **LAYOUT_BASE,
    )
    return fig


def plot_prompt_comparison(df: pd.DataFrame, metric: str, selected_models: List[str]) -> go.Figure:
    """Grouped bar: для кожної моделі — 3 промти поруч"""
    models = [m for m in selected_models if m in df['model'].values]
    fig = go.Figure()

    for prompt, color in PROMPT_COLORS.items():
        vals = []
        for m in models:
            recs = df[(df['model']==m) & (df['prompt']==prompt)]
            vals.append(recs[metric].mean() if not recs.empty else 0)
        fig.add_trace(go.Bar(
            name=PROMPT_UA.get(prompt, prompt),
            x=[m.split(':')[0] for m in models],
            y=vals,
            marker_color=color,
            text=[f"{v:,.1f}" for v in vals],
            textposition='outside',
            textfont=dict(size=9),
        ))

    fig.update_layout(
        barmode='group',
        title=f'{METRIC_UA.get(metric, metric)} — порівняння промтів по моделях',
        xaxis=dict(title='Модель', tickangle=-30, **AXIS_STYLE),
        yaxis=dict(title=METRIC_UA.get(metric, metric), **AXIS_STYLE),
        height=480, **LAYOUT_BASE,
    )
    return fig


def plot_config_lines(df: pd.DataFrame, metric: str) -> go.Figure:
    """Лінійний графік: вісь X = моделі, лінії = конфіги — показує тренд"""
    models = sorted(df['model'].unique(), key=lambda m: df[df['model']==m][metric].mean(), reverse=True)
    fig = go.Figure()

    for cfg, color in CONFIG_COLORS.items():
        vals = []
        for m in models:
            recs = df[(df['model']==m) & (df['config_label']==cfg)]
            vals.append(recs[metric].mean() if not recs.empty else 0)
        fig.add_trace(go.Scatter(
            x=[m.split(':')[0] for m in models],
            y=vals,
            mode='lines+markers',
            name=CONFIG_UA.get(cfg, cfg),
            line=dict(color=color, width=2),
            marker=dict(size=9, color=color),
        ))

    fig.update_layout(
        title=f'{METRIC_UA.get(metric, metric)} — тренд по конфігураціях',
        xaxis=dict(title='Модель (сортовано за середнім)', **AXIS_STYLE),
        yaxis=dict(title=METRIC_UA.get(metric, metric), **AXIS_STYLE),
        height=420, **LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# МОДУЛЬ 8 — МАТРИЦЯ МОДЕЛЬ × КОНФІГ × ПРОМТ
# ══════════════════════════════════════════════════════════════════════════════

def plot_full_matrix(df: pd.DataFrame, metric: str) -> go.Figure:
    """Теплова карта: рядки = модель×конфіг, стовпці = промт"""
    rows = []
    row_labels = []
    col_labels = [PROMPT_UA.get(p,p) for p in sorted(df['prompt'].unique())]
    prompts = sorted(df['prompt'].unique())

    for model in sorted(df['model'].unique()):
        for cfg in ['Standard', 'Creative', 'Precise']:
            row = []
            for prompt in prompts:
                recs = df[(df['model']==model)&(df['config_label']==cfg)&(df['prompt']==prompt)]
                row.append(round(recs[metric].mean(),1) if not recs.empty else 0)
            rows.append(row)
            row_labels.append(f"{model.split(':')[0]} / {CONFIG_UA.get(cfg,cfg)}")

    fig = go.Figure(data=go.Heatmap(
        z=rows, x=col_labels, y=row_labels,
        colorscale='RdYlGn',
        text=[[f"{v:,.0f}" if v>=10 else f"{v:.2f}" for v in row] for row in rows],
        texttemplate='%{text}', textfont=dict(size=9),
        hovertemplate='%{y}<br>%{x}<br>%{z}<extra></extra>',
    ))

    fig.update_layout(
        title=f'Повна матриця: {METRIC_UA.get(metric,metric)} для всіх комбінацій',
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(tickfont=dict(size=9), **AXIS_STYLE),
        height=max(500, len(row_labels)*28 + 100),
        **LAYOUT_BASE,
    )
    return fig


def config_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Зведена таблиця всіх 78 записів"""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'Модель':        r['model'].split(':')[0],
            'Конфігурація':  CONFIG_UA.get(r['config_label'], r['config_label']),
            'Промт':         PROMPT_UA.get(r['prompt'], r['prompt']),
            'Символів':      f"{r['char_count']:,}",
            'TTR':           f"{r['ttr']:.3f}",
            'Херданс C':     f"{r['herdan_c']:.3f}",
            'Сер. речення':  f"{r['avg_sentence_len']:.1f}",
            'Дискурс/100сл': f"{r['discourse_per_100w']:.2f}",
            'Швидкість TPS': f"{r['tokens_per_sec']:.0f}",
            'Зупинка':       r['done_reason'],
        })
    return pd.DataFrame(rows)


def build_app():
    default_json = 'text_analysis_results.json'
    default_zip  = 'generated_texts.zip'

    df0 = load_json(default_json) if Path(default_json).exists() else None
    if Path(default_zip).exists():
        n = load_texts_from_zip(default_zip)
        print(f'[+] Завантажено {n} текстових файлів з {default_zip}')

    theme = gr.themes.Base(
        primary_hue='teal',
        neutral_hue='slate',
        font=gr.themes.GoogleFont('JetBrains Mono'),
    )
    with gr.Blocks(title='Аналіз вихідних текстів LLM', theme=theme) as app:

        gr.HTML("""
        <div style="padding:16px 0 12px;border-bottom:1px solid #222;margin-bottom:16px">
          <h1 style="margin:0;font-size:22px;color:#2A9D8F;letter-spacing:2px">
            ⬡ АНАЛІЗ ПАРАМЕТРІВ ВИХІДНИХ ТЕКСТІВ LLM
          </h1>
          <p style="margin:4px 0 0;color:#555;font-size:12px">
            Бакалаврська робота — Пенцак Андрій, ЛНУ ім. Івана Франка, 2026 |
            Завантажте власний JSON для аналізу будь-яких LLM
          </p>
        </div>""")

        with gr.Row():
            json_file = gr.File(label='📄 JSON з результатами', file_types=['.json'],
                                value=default_json if df0 is not None else None)
            zip_file  = gr.File(label='🗜 ZIP з текстами (generated_texts.zip)',
                                file_types=['.zip'],
                                value=default_zip if Path(default_zip).exists() else None)
            load_btn  = gr.Button('Завантажити', variant='primary', scale=0)

        status = gr.HTML(
            value=summary_html(df0) if df0 is not None else
                  "<p style='color:#666;padding:8px'>Завантажте JSON файл для початку</p>")
        df_state = gr.State(value=df0)

        def on_load(jf, zf, _):
            if jf is None:
                return _, "<p style='color:#E63946'>Файл не обрано</p>"
            df = load_json(jf.name)
            if df is None:
                return _, "<p style='color:#E63946'>Помилка завантаження JSON</p>"
            if zf is not None:
                n = load_texts_from_zip(zf.name)
                print(f'[+] {n} текстів завантажено')
            return df, summary_html(df)

        load_btn.click(on_load, [json_file, zip_file, df_state], [df_state, status])

        with gr.Tabs():

            # ── Вкладка 1: Закон Зіпфа ───────────────────────────────────────
            with gr.Tab('① Закон Зіпфа'):
                gr.Markdown("""
**Закон Зіпфа** (1935): частота слова обернено пропорційна його рангу.
`f(r) ∝ 1/r^α` — ідеальний людський текст: **α ≈ 1.0**. Відхилення LLM?

**Закон Хіпса** (1978): зростання словника `V(n) ≈ K·n^β`, ідеал **β ≈ 0.5**
*(Курс: лабораторна №02, №04)*
""")
                zipf_sel = gr.CheckboxGroup(
                    label='Оберіть моделі',
                    choices=get_models(df0) if df0 is not None else [],
                    value=get_models(df0)[:5] if df0 is not None else [])
                zipf_plot  = gr.Plot()
                zipf_table = gr.Dataframe(label='Статистика Зіпфа та Хіпса')
                gr.Button('▶ Запустити аналіз Зіпфа', variant='primary').click(
                    lambda sel, df: (plot_zipf(df,sel), zipf_stats_table(df,sel))
                        if df is not None else (None, pd.DataFrame()),
                    [zipf_sel, df_state], [zipf_plot, zipf_table])

            # ── Вкладка 2: Відбиток моделі ───────────────────────────────────
            with gr.Tab('② Відбиток моделі'):
                gr.Markdown("""
**Відбиток моделі**: кожна LLM має унікальний стилістичний профіль за 8 вимірами.
Радарна діаграма — нормалізовані значення для візуального порівняння.
Теплова карта — де моделі найбільше схожі та відрізняються.
*(Курс: лабораторна №21 — кореляції)*
""")
                fp_sel    = gr.CheckboxGroup(
                    label='Оберіть моделі',
                    choices=get_models(df0) if df0 is not None else [],
                    value=get_models(df0) if df0 is not None else [])
                fp_radar  = gr.Plot(label='Радар — стильовий профіль')
                fp_heat   = gr.Plot(label='Теплова карта')
                gr.Button('▶ Побудувати відбитки', variant='primary').click(
                    lambda sel, df: (plot_fingerprint(df, sel or get_models(df)),
                                     plot_heatmap(df))
                        if df is not None else (None, None),
                    [fp_sel, df_state], [fp_radar, fp_heat])

            # ── Вкладка 3: Парадокс якість–кількість ────────────────────────
            with gr.Tab('③ Парадокс якість–кількість'):
                gr.Markdown("""
**Парадокс**: більше тексту → менша лексична різноманітність (нижчий TTR).
Чи це універсальний закон, чи специфіка конкретної моделі?
**Херданс C** — незалежна від довжини альтернатива TTR.
*(Курс: лабораторна №04 — Хіпс, TTR)*
""")
                qq_sel     = gr.CheckboxGroup(
                    label='Оберіть моделі',
                    choices=get_models(df0) if df0 is not None else [],
                    value=get_models(df0) if df0 is not None else [])
                qq_scatter = gr.Plot(label='TTR vs Кількість символів')
                qq_box     = gr.Plot(label='TTR vs Херданс C')
                gr.Button('▶ Аналізувати парадокс', variant='primary').click(
                    lambda sel, df: (plot_ttr_scatter(df, sel or get_models(df)),
                                     plot_herdan_box(df))
                        if df is not None else (None, None),
                    [qq_sel, df_state], [qq_scatter, qq_box])

            # ── Вкладка 4: Ефект промту vs моделі ───────────────────────────
            with gr.Tab('④ Ефект промту vs моделі'):
                gr.Markdown("""
**Ключове питання**: що більше впливає на обсяг тексту — яку **модель** обрати,
чи **як сформулювати** промт?

**η² (ета-квадрат)** — частка дисперсії, яку пояснює кожен фактор (0→1).
*(Курс: лабораторна №16 — характеристики повторень)*
""")
                pme_plot  = gr.Plot()
                pme_table = gr.Dataframe(label='Статистика за типом промту')
                gr.Button('▶ Розкласти ефекти', variant='primary').click(
                    lambda df: (plot_effects(df), prompt_table(df))
                        if df is not None else (None, pd.DataFrame()),
                    [df_state], [pme_plot, pme_table])

            # ── Вкладка 5: Карта ефективності ───────────────────────────────
            with gr.Tab('⑤ Карта ефективності'):
                gr.Markdown("""
**Карта ефективності**: інтерактивний бульбашковий графік — оберіть будь-які
3 метрики для осей X, Y та розміру бульбашки.
Знайдіть оптимальну модель для вашого завдання.
""")
                with gr.Row():
                    eff_x  = gr.Dropdown(label='Вісь X', choices=METRICS_UA, value='tokens_per_sec')
                    eff_y  = gr.Dropdown(label='Вісь Y', choices=METRICS_UA, value='char_count')
                    eff_sz = gr.Dropdown(label='Розмір бульбашки', choices=METRICS_UA, value='ttr')
                eff_sel  = gr.CheckboxGroup(
                    label='Оберіть моделі',
                    choices=get_models(df0) if df0 is not None else [],
                    value=get_models(df0) if df0 is not None else [])
                eff_plot = gr.Plot()
                gr.Button('▶ Побудувати карту', variant='primary').click(
                    lambda x,y,sz,sel,df: plot_efficiency(df,sel or get_models(df),x,y,sz)
                        if df is not None else None,
                    [eff_x, eff_y, eff_sz, eff_sel, df_state], [eff_plot])

            # ── Вкладка 6: Розподіл речень ───────────────────────────────────
            with gr.Tab('⑥ Розподіл речень'):
                gr.Markdown("""
**Розподіл довжин речень** (Лаб. №15 — аналіз довжин слів та речень).
Скрипковий графік: форма розподілу, середнє, медіана, дисперсія, асиметрія.
Довгий хвіст → складний академічний стиль. Короткі речення → розмовний стиль.
""")
                sd_sel  = gr.CheckboxGroup(
                    label='Оберіть моделі',
                    choices=get_models(df0) if df0 is not None else [],
                    value=get_models(df0)[:5] if df0 is not None else [])
                sd_plot = gr.Plot()
                gr.Button('▶ Побудувати розподіли', variant='primary').click(
                    lambda sel, df: plot_violin(df, sel or get_models(df))
                        if df is not None else None,
                    [sd_sel, df_state], [sd_plot])

            # ── Вкладка 7: Порівняння конфігурацій ──────────────────────────
            with gr.Tab('⑦ Конфіг та промт'):
                gr.Markdown("""
**Ключове питання**: як змінюються параметри тексту при різних конфігураціях
(Стандартний / Творчий / Точний) та різних промтах для **однієї і тієї ж моделі**?
Це показує наскільки можна «налаштувати» модель без зміни ваг.
""")
                with gr.Row():
                    cfg_metric = gr.Dropdown(
                        label='Метрика для порівняння',
                        choices=METRICS_UA,
                        value='char_count')
                    cfg_model_sel = gr.CheckboxGroup(
                        label='Моделі',
                        choices=get_models(df0) if df0 is not None else [],
                        value=get_models(df0) if df0 is not None else [])

                cfg_bar   = gr.Plot(label='Grouped bar — конфігурації')
                cfg_pbar  = gr.Plot(label='Grouped bar — промти')
                cfg_lines = gr.Plot(label='Лінійний тренд по конфігураціях')

                gr.Button('▶ Порівняти конфігурації та промти', variant='primary').click(
                    lambda metric, sel, df: (
                        plot_config_comparison(df, metric, sel or get_models(df)),
                        plot_prompt_comparison(df, metric, sel or get_models(df)),
                        plot_config_lines(df, metric),
                    ) if df is not None else (None, None, None),
                    [cfg_metric, cfg_model_sel, df_state],
                    [cfg_bar, cfg_pbar, cfg_lines])

            # ── Вкладка 8: Повна матриця ─────────────────────────────────────
            with gr.Tab('⑧ Повна матриця'):
                gr.Markdown("""
**Повна матриця**: теплова карта всіх **78 комбінацій** (9 моделей × 3 конфіги × 3 промти).
Дозволяє знайти найкращу та найгіршу комбінацію для кожної метрики.
Кольорова шкала: зелений = вище, червоний = нижче.
""")
                matrix_metric = gr.Dropdown(
                    label='Метрика',
                    choices=METRICS_UA,
                    value='char_count')
                matrix_plot = gr.Plot(label='Матриця модель × конфіг × промт')
                matrix_table = gr.Dataframe(
                    label='Всі 78 записів',
                    wrap=True)

                gr.Button('▶ Побудувати матрицю', variant='primary').click(
                    lambda metric, df: (
                        plot_full_matrix(df, metric),
                        config_stats_table(df),
                    ) if df is not None else (None, pd.DataFrame()),
                    [matrix_metric, df_state],
                    [matrix_plot, matrix_table])

            # ── Вкладка 9: Експорт ───────────────────────────────────────────
            with gr.Tab('⑨ Експорт'):
                gr.Markdown('Завантажте оброблену статистику для дипломної роботи.')
                export_btn  = gr.Button('Згенерувати CSV зведення', variant='primary')
                export_file = gr.File(label='Завантажити CSV')

                def do_export(df):
                    if df is None: return None
                    path = '/tmp/llm_export.csv'
                    cols = [c for c in df.columns if c not in ('full_text','txt_file','color')]
                    df[cols].to_csv(path, index=False)
                    return path

                export_btn.click(do_export, [df_state], [export_file])

                gr.Markdown("""
---
**Про інструмент**

Розроблено для бакалаврської роботи *«Аналіз параметрів вихідних текстів
у сучасних генеративних мовних моделях»* — Пенцак Андрій, ЛНУ ім. Франка, 2026.

**Методи:**
- **Закон Зіпфа** (1935) — ранг-частотний розподіл слів
- **Закон Хіпса** (1978) — зростання словника
- **TTR та Херданс C** — лексична різноманітність
- **η²** — розклад дисперсії (силовий аналіз)
- **Скрипкові графіки** — розподіл довжин речень (лаб. №15)
- **Радарне профілювання** — багатовимірний стильовий профіль

**Для власних даних:**
1. Запустіть `text_analysis_v3.py` на своєму Ollama
2. Завантажте `text_analysis_results.json` та `generated_texts.zip`
3. Всі аналізи запускаються автоматично
""")

        # Авто-запуск при старті якщо є дані
        if df0 is not None:
            app.load(
                lambda df: (
                    plot_fingerprint(df, get_models(df)),
                    plot_heatmap(df),
                    plot_ttr_scatter(df, get_models(df)),
                    plot_herdan_box(df),
                    plot_effects(df),
                    prompt_table(df),
                    plot_efficiency(df, get_models(df), 'tokens_per_sec', 'char_count', 'ttr'),
                ) if df is not None else (None,)*7,
                inputs=[df_state],
                outputs=[fp_radar, fp_heat, qq_scatter, qq_box,
                         pme_plot, pme_table, eff_plot],
            )

    return app


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data',  default='text_analysis_results.json')
    p.add_argument('--texts', default='generated_texts.zip')
    p.add_argument('--port',  type=int, default=7860)
    p.add_argument('--share', action='store_true')
    p.add_argument('--host',  default='0.0.0.0')
    args = p.parse_args()

    print('='*60)
    print('  АНАЛІЗ ВИХІДНИХ ТЕКСТІВ LLM — ДАШБОРД')
    print(f'  Дані: {args.data}')
    print(f'  Тексти: {args.texts}')
    print(f'  URL: http://localhost:{args.port}')
    print('='*60)

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port,
               share=args.share, show_error=True)
