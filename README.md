# 📊 Аналіз параметрів вихідних текстів LLM

> Інструментарій для автоматизованого порівняльного аналізу 20 лінгвістичних і статистичних параметрів вихідних текстів дев'яти відкритих великих мовних моделей (LLM) з інтерактивним аналітичним дашбордом.

---

## 👤 Автор

- **ПІБ**: Пенцак Андрій Ігорович
- **Група**: ФЕІ-43
- **Керівник**: Свелеба Сергій Антонович, проф.
- **Університет**: Львівський національний університет імені Івана Франка, 2026

---

## 📌 Загальна інформація

- **Тип проєкту**: Дослідницький інструментарій (Python CLI + Gradio Web UI)
- **Мова програмування**: Python 3.12
- **Бібліотеки**: requests, gradio, plotly, pandas, scipy, numpy
- **Платформа запуску LLM**: [Ollama](https://ollama.com) v0.24+
- **Рекомендоване обладнання**: GPU з ≥ 8 ГБ VRAM (RTX 3080+) або хмарний GPU

---

## 🧠 Функціонал проєкту

- 🤖 **Автоматизована генерація** текстів від 9 LLM через Ollama REST API (потоковий режим)
- 📐 **Обчислення 20 параметрів** тексту: обсяг, TTR, Херданс C, закон Зіпфа, закон Хіпса, дискурсивні маркери, лексична густота тощо
- 💾 **Збереження у трьох форматах**: JSON (повні метадані), CSV (таблиця параметрів), TXT (повні тексти)
- 📊 **Інтерактивний дашборд** (Gradio + Plotly) з 9 вкладками аналізу
- 🔄 **Захист від втрати даних**: кожен запис зберігається одразу після генерації
- 🌐 **Публічний доступ**: дашборд підтримує `--share` для Gradio-посилання

---

## 🧱 Структура репозиторію

| Файл / Папка | Призначення |
|---|---|
| `text_analysis_v3.py` | Основний pipeline: генерація + аналіз 78 записів × 20 параметрів |
| `dashboard.py` | Інтерактивний Gradio дашборд з 9 вкладками |
| `text_analysis_results.json` | Результати: 78 записів з усіма параметрами |
| `text_analysis_results.csv` | Те саме у табличному форматі CSV |
| `generated_texts/` | 78 TXT файлів із повними текстами (назва: `model__config__prompt.txt`) |
| `summary_per_model.csv` | Середні значення параметрів по моделях |
| `summary_per_config.csv` | Середні значення по конфігураціях генерації |
| `summary_per_prompt.csv` | Середні значення по типах промтів |
| `Modelfile` | Шаблон Ollama Modelfile для кастомних налаштувань |

---

## ▶️ Як запустити — варіант 1: власна генерація

### 1. Встановлення інструментів

**Python 3.10+** та **pip**:
```bash
python3 --version   # має бути >= 3.10
pip install requests gradio plotly pandas scipy numpy
```

**Ollama** (локальний сервер для LLM):
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Завантажте інсталятор з https://ollama.com/download
```

### 2. Клонування репозиторію

```bash
git clone https://github.com/andriy59b/diplom_text_analy.git
cd diplom_text_analy
```

### 3. Завантаження моделей

```bash
ollama serve &   # запустити сервер у фоні

# Завантажити всі 9 моделей (~65 ГБ разом):
ollama pull gemma4:e4b
ollama pull gemma3:12b
ollama pull qwen3:14b
ollama pull qwen2.5:3b
ollama pull granite4.1:8b
ollama pull phi4:14b
ollama pull deepseek-r1:1.5b
ollama pull mistral-nemo:12b
ollama pull gpt-oss:20b

# Перевірити що всі завантажені:
ollama list
```

> ⚠️ **Увага**: Для завантаження потрібно ~65–70 ГБ вільного місця на диску та ~8–32 ГБ VRAM залежно від моделі. Рекомендовано хмарний GPU (Vast.ai, RunPod) з RTX 3090 або новіше.

### 4. Запуск збору даних

```bash
python text_analysis_v3.py
```

Скрипт автоматично:
- Перевіряє які моделі доступні в Ollama
- Запускає 9 × 3 × 3 = 78 генерацій (пропускає вже виконані)
- Зберігає кожен текст одразу після генерації (`generated_texts/*.txt`)
- Оновлює `text_analysis_results.json` та `text_analysis_results.csv` після кожного запису

**Орієнтовний час**: ~1–3 години на RTX 5090, ~6–12 годин на RTX 3090.

---

## 📊 Як запустити дашборд — варіант 2: тільки аналіз готових даних

Якщо не хочете запускати генерацію — у репозиторії вже є `text_analysis_results.json` з 78 записами та `generated_texts.zip` з повними текстами.

```bash
# Клонувати репозиторій
git clone https://github.com/andriy59b/diplom_text_analy.git
cd diplom_text_analy

# Встановити залежності
pip install gradio plotly pandas scipy numpy

# Запустити дашборд з готовими даними
python dashboard.py --data text_analysis_results.json --texts generated_texts.zip

# Або з публічним посиланням:
python dashboard.py --data text_analysis_results.json --texts generated_texts.zip --share
```

Дашборд відкриється на **http://localhost:7860**

При `--share` у терміналі з'явиться публічне посилання вигляду `https://xxxxx.gradio.live` (дійсне 72 год).

---

## 🖥️ Аргументи командного рядка

### `text_analysis_v3.py`

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| *(немає)* | — | Запускається без аргументів, читає конфігурацію з коду |

### `dashboard.py`

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| `--data` | `text_analysis_results.json` | Шлях до JSON з результатами |
| `--texts` | `generated_texts.zip` | Шлях до ZIP з TXT файлами текстів |
| `--port` | `7860` | Порт для веб-сервера |
| `--host` | `0.0.0.0` | Хост для прослуховування |
| `--share` | `False` | Створити публічне Gradio-посилання |

---

## 📐 Параметри, що обчислюються (20 штук)

| Параметр | Опис |
|---|---|
| `char_count` | Кількість символів |
| `word_count` | Кількість слів |
| `sentence_count` | Кількість речень |
| `paragraph_count` | Кількість абзаців |
| `unique_words` | Кількість унікальних слів |
| `ttr` | Type-Token Ratio — лексична різноманітність |
| `herdan_c` | Херданс C — TTR незалежний від довжини |
| `avg_sentence_len` | Середня довжина речення (слів) |
| `std_sentence_len` | Стандартне відхилення довжин речень |
| `avg_word_len` | Середня довжина слова (символів) |
| `avg_para_len` | Середня довжина абзацу (речень) |
| `discourse_markers` | Кількість дискурсивних маркерів |
| `discourse_per_100w` | Маркерів на 100 слів (нормалізовано) |
| `lexical_density` | Частка повнозначних слів |
| `long_sentences_pct` | % речень довших 30 слів |
| `short_sentences_pct` | % речень коротших 5 слів |
| `repeat_word_pct` | % унікальних слів що зустрічаються > 3 разів |
| `tokens_generated` | Кількість згенерованих токенів |
| `done_reason` | Причина зупинки: `stop` або `length` |
| `tokens_per_sec` | Швидкість генерації (токенів/сек) |

---

## 📊 Вкладки дашборду

| # | Вкладка | Метод аналізу |
|---|---|---|
| ① | Закон Зіпфа | Ранг-частотний розподіл + закон Хіпса |
| ② | Відбиток моделі | Радарна діаграма + теплова карта профілів |
| ③ | Парадокс якість–кількість | TTR vs обсяг + степенева апроксимація |
| ④ | Ефект промту vs моделі | η²-аналіз дисперсії |
| ⑤ | Карта ефективності | Бульбашковий графік (TPS × обсяг × TTR) |
| ⑥ | Розподіл речень | Violin plot довжин речень |
| ⑦ | Конфіг та промт | Grouped bar chart + лінійний тренд |
| ⑧ | Повна матриця | Теплова карта 78 комбінацій |
| ⑨ | Експорт | Вивантаження CSV |

---

## 🧪 Проблеми і рішення

| Проблема | Рішення |
|---|---|
| `Connection refused` при зверненні до Ollama | Запустити `ollama serve` або перевірити що процес активний |
| `model not found` | Виконати `ollama pull <назва_моделі>` |
| Дашборд не відкривається | Перевірити що порт 7860 вільний: `lsof -i :7860` |
| Вкладки ① і ⑥ порожні | Передати `--texts generated_texts.zip` (потрібні повні тексти) |
| `CUDA out of memory` | Зменшити `num_ctx` до 32768 у конфігурації, або використовувати менші моделі |
| Повільна генерація | Нормально для CPU-режиму; GPU прискорює в 10–50 разів |
| `ImportError: No module named ...` | Виконати `pip install gradio plotly pandas scipy numpy` |

---

## 🔬 Досліджувані моделі

| Модель | Розробник | Параметрів |
|---|---|---|
| `gemma4:e4b` | Google | ~27B (MoE) |
| `gemma3:12b` | Google | 12B |
| `qwen3:14b` | Alibaba | 14B |
| `qwen2.5:3b` | Alibaba | 3B |
| `granite4.1:8b` | IBM | 8B |
| `phi4:14b` | Microsoft | 14B |
| `deepseek-r1:1.5b` | DeepSeek | 1.5B |
| `mistral-nemo:12b` | Mistral AI | 12B |
| `gpt-oss:20b` | OpenAI | 20B |

---

## 📷 Скріншоти

> Скріншоти дашборду доступні у папці `/screenshots/` репозиторію.

- Головна сторінка з підсумковими картками
- Вкладка ① Закон Зіпфа
- Вкладка ② Відбиток моделі
- Вкладка ③ Парадокс якість–кількість
- Вкладка ⑧ Повна матриця 78 комбінацій

---

## 🧾 Використані джерела

- [Ollama documentation](https://ollama.com/docs)
- [Gradio documentation](https://www.gradio.app/docs)
- [Plotly Python](https://plotly.com/python/)
- Zipf G.K. The Psycho-Biology of Language. 1935.
- Heaps H.S. Information Retrieval. 1978.
- Herdan G. Type-token mathematics. 1960.
- Vaswani et al. Attention Is All You Need. NeurIPS 2017.

---

## 📄 Ліцензія

MIT License — вільне використання з посиланням на автора.
