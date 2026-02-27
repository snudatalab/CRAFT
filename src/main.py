"""
main.py — CRAFT single-run entry point.

Paper: "Accurate Stock Movement Prediction via Centroid-based Randomness Smoothing"

Runs a single (region, test period, hyperparameter) combination.
Not a grid search — use grid_search.py for systematic sweeps.

Execution flow:
  1. Data loading (fetch_data) -> price DataFrame
  2. Feature construction (compute_base_features) -> tokens + SVD + global ctx
  3. Sequence generation (build_sequences) -> CRAFT input format
  4. Date-based splitting (split_by_dates) -> train / valid / test
  5. Training and evaluation (run_experiment) -> ACC, MCC, ASR, RMDD, AVol

Date split rules (paper Table 2):
  - test  : [test_start, test_end]
  - valid : test_start - 3 months ~ test_start - 1 day
  - train : ~ valid_start - 1 day

Usage:
    python -m src.main --region USA --test_start 2024-01-01 --test_end 2024-06-30
    python -m src.main --region CHN --test_start 2020-01-01 --test_end 2020-03-31 \\
                       --w 10 --k 200 --seq_len 10 --d_model 128 --n_heads 4
"""

import argparse
import time
import pandas as pd

from src.preprocess import (
    fetch_data, compute_base_features, build_sequences, split_by_dates,
)
from src.engine import run_experiment


REGION_CODES = ["USA", "CHN", "JPN", "EUR"]


def _resolve_splits(test_start: str, test_end: str):
    """Compute train/valid boundary dates from the test period.

    Paper Table 2 split rules:
      - valid : test_start - 3 months ~ test_start - 1 day
      - train : ~ valid_start - 1 day

    Args:
        test_start (str): Test period start date (YYYY-MM-DD).
        test_end (str): Test period end date.

    Returns:
        tuple: (train_end, valid_start, valid_end) as date strings.
    """
    ts = pd.to_datetime(test_start)
    valid_start = ts - pd.DateOffset(months=3)
    valid_end = ts - pd.Timedelta(days=1)
    train_end = valid_start - pd.Timedelta(days=1)
    return str(train_end.date()), str(valid_start.date()), str(valid_end.date())


def main():
    """CRAFT single run: parse args -> load data -> build features -> train/eval.

    Key arguments (paper Hyperparameters section):
        --region       : Region code ("USA", "CHN", "JPN", "EUR")
        --test_start   : Test start date
        --test_end     : Test end date
        --w            : Price window size (paper: {5, 10, 20}). Default 10.
        --k            : Number of centroids (paper: 200 optimal). Default 200.
        --seq_len      : Token sequence length L (paper: {5, 10, 20}). Default 10.
        --d_model      : Hidden dimension d (paper: {64, 128, 256}). Default 64.
        --n_heads      : Attention heads (paper: {1, 2, 4}). Default 2.
        --n_layers     : Transformer layers (paper: {1, 2, 4}). Default 2.
        --lambda_stock : L_stock weight (paper: {0.1, ..., 0.9}). Default 0.1.
        --lr           : Learning rate (paper: {0.01, 0.001, 0.0001}). Default 1e-3.
        --epochs       : Total epochs. Default 200.
        --seed         : Random seed (paper: 5 seeds 0-4). Default 42.
    """
    parser = argparse.ArgumentParser(description="CRAFT single run")
    parser.add_argument("--region", type=str, default="USA", choices=REGION_CODES)
    parser.add_argument("--test_start", type=str, default="2024-01-01")
    parser.add_argument("--test_end", type=str, default="2024-06-30")
    parser.add_argument("--w", type=int, default=10,
                        help="Price window size w (for relative returns)")
    parser.add_argument("--k", type=int, default=200,
                        help="k-means centroid count (paper: 200 optimal)")
    parser.add_argument("--seq_len", type=int, default=10,
                        help="Input token sequence length L")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_stock", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_embed", type=int, default=10,
                        help="SVD embedding dimension d_embed")
    parser.add_argument("--corr_window", type=int, default=60,
                        help="Correlation rolling window (trading days)")
    args = parser.parse_args()

    # Compute date split boundaries
    train_end, valid_start, valid_end = _resolve_splits(args.test_start, args.test_end)

    print(f"[Config] region={args.region}  test={args.test_start}~{args.test_end}")
    print(f"         train<=  {train_end}  valid={valid_start}~{valid_end}")
    print(f"         w={args.w}  k={args.k}  L={args.seq_len}  d={args.d_model}  "
          f"heads={args.n_heads}  layers={args.n_layers}  lambda={args.lambda_stock}")

    # 1) Data loading
    t0 = time.time()
    df = fetch_data(args.region)
    print(f"[Data] Fetched {len(df)} rows  ({time.time()-t0:.1f}s)")

    # 2) Feature construction: tokenisation + SVD + global context
    t0 = time.time()
    features = compute_base_features(
        df, w=args.w, k=args.k, d_embed=args.d_embed,
        corr_window=args.corr_window, train_end_date=train_end,
    )
    print(f"[Features] n_stocks={len(features['symbols'])}  "
          f"n_clusters={features['n_clusters']}  ({time.time()-t0:.1f}s)")

    # 3) Sequence generation
    seq_data = build_sequences(features, args.seq_len)
    print(f"[Sequences] total={len(seq_data['dates'])}")

    # 4) Date-based splitting
    train_d, valid_d, test_d = split_by_dates(
        seq_data, train_end, valid_end, args.test_start, args.test_end,
    )
    print(f"[Split] train={len(train_d['dates'])}  "
          f"valid={len(valid_d['dates'])}  test={len(test_d['dates'])}")

    if len(train_d["dates"]) == 0 or len(valid_d["dates"]) == 0:
        print("[ERROR] Insufficient data for train/valid split.")
        return

    if len(test_d["dates"]) == 0:
        print("[ERROR] No test data in the specified period.")
        return

    # 5) Training and evaluation
    results = run_experiment(
        train_data=train_d,
        valid_data=valid_d,
        test_data=test_d,
        n_clusters=features["n_clusters"],
        feat_dim=features["feat_dim"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lambda_stock=args.lambda_stock,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Test metrics by epoch checkpoint")
    print("-" * 60)
    for ep in sorted(results.keys()):
        m = results[ep]
        parts = ["epoch {:>4d}".format(ep),
                 "ACC {:.4f}".format(m.get("ACC", 0.0)),
                 "MCC {:.4f}".format(m.get("MCC", 0.0))]
        if "ASR" in m:
            parts.extend([
                "ASR {:.4f}".format(m["ASR"]),
                "RMDD {:.4f}".format(m["RMDD"]),
                "AVol {:.4f}".format(m["AVol"]),
            ])
        print("  " + " | ".join(parts))
    print("=" * 60)


if __name__ == "__main__":
    main()
