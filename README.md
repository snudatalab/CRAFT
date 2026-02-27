# CRAFT — Centroid-based Randomness Smoothing Approach for Stock Forecasting with Transformer

Implementation of the paper:  
**"Accurate Stock Movement Prediction via Centroid-based Randomness Smoothing"**

---

## Project Structure

```
project/
├── data/                    # ⬇ Download and place CSV files here (see below)
│   ├── USA.csv              # S&P 500
│   ├── CHN.csv              # CSI 300
│   ├── JPN.csv              # NIKKEI 225
│   └── EUR.csv              # EURO STOXX 50
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data loading, tokenisation, SVD embedding, global context
│   ├── engine.py            # CRAFT model, training loop, evaluation metrics
│   └── main.py              # CLI entry point
├── cache/                   # Auto-generated pickle caches (gitignored)
└── README.md
```

## Dataset Setup

The dataset is **not included** in this repository. Download it from Google Drive and place the CSV files into the `data/` directory before running the pipeline.

**Download link:**  
https://drive.google.com/file/d/1cdkMmwTDVox4raTdoxi8bLPVN-tjqo8y/view?usp=sharing

```bash
# 1. Create the data directory
mkdir -p data

# 2. Download USA.csv, CHN.csv, JPN.csv, EUR.csv from the link above

# 3. Place them so the layout looks like:
#    data/USA.csv
#    data/CHN.csv
#    data/JPN.csv
#    data/EUR.csv
```

### Data Format

Each CSV must contain the following columns (case-insensitive):

| Column   | Type   | Description              |
|----------|--------|--------------------------|
| `date`   | str    | Trading date (YYYY-MM-DD)|
| `symbol` | str    | Stock ticker             |
| `close`  | float  | Closing price            |
| `volume` | float  | Trading volume           |

Optional columns (`open`, `high`, `low`) are accepted but not required by the pipeline.  
Missing prices are forward/backward filled automatically.

---

## Pipeline Overview

The pipeline follows Figure 1 of the paper:

```
CSV Data
  │
  ▼
┌─────────────────────────────────────────────┐
│  preprocess.py                              │
│                                             │
│  1. Relative return vectors  x_{i,t}        │  ← Token Generation
│  2. k-means centroid tokens  z_{i,t}        │
│  3. SVD stock embeddings     e_{i,t}        │  ← Eq. (1)-(2)
│  4. Global context           g_{i,t}        │  ← Eq. (3)
│  5. Sequence construction  (L, N, F)        │
│  6. Date-based train/valid/test split       │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  engine.py  — CRAFT Model                   │
│                                             │
│  Linear projection  [z ‖ g] → d_model       │
│  Time-axis masked self-attention             │  ← Eq. (4)-(5)
│  Token prediction head (CE over k centroids) │  ← Eq. (6)
│  Stock-axis multi-head self-attention        │  ← Eq. (7)-(8)
│  Movement prediction head (up/down)          │
│                                             │
│  Loss: L_total = L_time + λ · L_stock       │
└─────────────────────────────────────────────┘
  │
  ▼
  Metrics: ACC, MCC, ASR, RMDD, AVol
```

### Two-Level Caching

- **Level 1 (region):** Price matrix, daily returns, relative returns, SVD embeddings.  
  Independent of train/test split — computed once per region.
- **Level 2 (period):** k-means centroids, token embeddings, global context.  
  Depends on `train_end` date (prevents look-ahead bias).

Cache files are stored in `cache/` as pickle and reused across runs.

---

## Usage

```bash
python -m src.main \
    --region USA \
    --test_start 2024-01-01 \
    --test_end 2024-06-30 \
    --w 10 --k 200 --seq_len 10 \
    --d_model 64 --n_heads 2 --n_layers 2 \
    --lambda_stock 0.1 --lr 1e-3 \
    --epochs 200 --seed 42
```

### Date Split Rules (Paper Table 2)

Given a test period `[test_start, test_end]`:

| Split      | Range                                          |
|------------|------------------------------------------------|
| **Test**   | `test_start` ≤ date ≤ `test_end`              |
| **Valid**  | `test_start − 3 months` ≤ date < `test_start` |
| **Train**  | date < `valid_start`                           |

---

## Hyperparameters

| Parameter      | Default | Paper Reference                     |
|----------------|---------|-------------------------------------|
| `w`            | 10      | Price window size ({5, 10, 20})     |
| `k`            | 200     | Centroid count (Figure 2 optimum)   |
| `seq_len`      | 10      | Sequence length ℓ ({5, 10, 20})    |
| `d_model`      | 64      | Hidden dimension d ({64, 128, 256}) |
| `n_heads`      | 2       | Attention heads ({1, 2, 4})         |
| `n_layers`     | 2       | Transformer layers ({1, 2, 4})      |
| `lambda_stock` | 0.1     | L_stock weight λ ({0.1, …, 0.9})   |
| `lr`           | 1e-3    | Adam learning rate                  |
| `epochs`       | 200     | Total training epochs               |
| `batch_size`   | 32      |                                     |
| `dropout`      | 0.1     |                                     |
| `d_embed`      | 10      | SVD embedding dimension             |
| `corr_window`  | 60      | Correlation window (~3 months)      |

---

## Evaluation Metrics

| Metric | Full Name                     | Direction | Formula                                    |
|--------|-------------------------------|-----------|--------------------------------------------|
| ACC    | Accuracy                      | ↑         | Correct predictions / Total                |
| MCC    | Matthews Correlation Coeff.   | ↑         | (tp·tn − fp·fn) / √((tp+fp)(tp+fn)(tn+fp)(tn+fn)) |
| ASR    | Annualised Sharpe Ratio       | ↑         | (μ/σ) × √252                              |
| RMDD   | Relative Max Drawdown         | ↓         | Max peak-to-trough loss                    |
| AVol   | Annualised Volatility         | ↓         | σ × √252                                  |

Investment simulation follows a daily-rebalanced equal-weight long/short strategy:  
predicted **up → long (+1)**, predicted **down → short (−1)**.

---

## Requirements

```
numpy
pandas
scikit-learn
torch >= 1.13
```

Multi-GPU training is supported via `nn.DataParallel` (no distributed setup required).

---

## Module Reference

### `src/preprocess.py`

| Function                  | Purpose                                                  |
|---------------------------|----------------------------------------------------------|
| `fetch_data()`            | Load regional CSV into DataFrame                         |
| `cache_region_base()`     | Level-1 cache: prices, returns, relative returns, SVD    |
| `build_period_features()` | Level-2: k-means tokenisation + global context           |
| `compute_base_features()` | Single-call wrapper (L1 + L2 combined)                   |
| `build_sequences()`       | Construct (L, N, F) input sequences and targets          |
| `split_by_dates()`        | Date-based train/valid/test split                        |

### `src/engine.py`

| Function / Class    | Purpose                                                    |
|---------------------|------------------------------------------------------------|
| `CRAFT`             | Full model: time-axis + stock-axis attention + heads       |
| `run_experiment()`  | Train + evaluate at checkpoint epochs                      |
| `evaluate_full()`   | Compute ACC, MCC, ASR, RMDD, AVol on test set             |
| `backtest_stats()`  | ASR / RMDD / AVol from daily return sequence               |
| `compute_mcc()`     | Matthews Correlation Coefficient                           |

### `src/main.py`

CLI entry point. Calls `fetch_data → compute_base_features → build_sequences → split_by_dates → run_experiment`.
