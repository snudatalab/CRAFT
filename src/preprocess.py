"""
preprocess.py — CRAFT data loading, centroid-based tokenisation, SVD stock
                embedding, and global context construction.

Paper: "Accurate Stock Movement Prediction via Centroid-based Randomness Smoothing"

Core preprocessing stages:
  I1. Token Generation:
      - Relative return vectors x_{i,t} in R^{w-1}
      - k-means clustering to produce centroid tokens z_{i,t} = v_{phi(x_{i,t})}
  I2. Related Context Aggregation via Dynamic Stock Embedding:
      - SVD-based temporal stock embeddings e_{i,t} (Eq. 1-2)
      - Global context g_{i,t} = sum_j alpha_{ij,t} * z_{j,t} (Eq. 3)
  Sequence construction and date-based data splitting.

Two-level caching strategy:
  Level 1 (region): Price matrix, daily returns, relative returns, SVD embeddings.
                    Computed once, independent of train/test split.
  Level 2 (period): k-means centroids, token embeddings, global context.
                    Depends on train_end date (no look-ahead).

Tensor dimension conventions:
    T = total trading days
    N = number of stocks (universe size)
    w = price window size (paper: w in {5, 10, 20})
    F = w-1 (relative return vector dimension)
    k = number of centroids (paper: k=200 optimal, Figure 2)
    d_embed = SVD embedding dimension (default 10)
    L = sequence length (input token sequence length)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Data directory and region-to-file mapping
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

REGION_FILE = {
    "USA": "USA.csv",
    "CHN": "CHN.csv",
    "JPN": "JPN.csv",
    "EUR": "EUR.csv",
}


# ---------------------------------------------------------------------------
# Data loading (CSV-based)
# ---------------------------------------------------------------------------

def fetch_data(region, data_dir=None, cache_dir="cache"):
    """Load price data from a regional CSV file.

    Paper Table 2: four markets (USA, CHN, JPN, EUR).

    The CSV is expected to have columns: date, symbol, open, high, low,
    close, volume (and possibly others). Rows are sorted by symbol and date.

    Args:
        region (str): Region code — one of "USA", "CHN", "JPN", "EUR".
        data_dir (str or None): Override for data directory.
            None uses the auto-detected ``data/`` folder.
        cache_dir (str): Pickle cache directory. Default "cache".

    Returns:
        pd.DataFrame: Columns include date, symbol, close, volume, etc.
            Sorted by [symbol, date].
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "raw_{}.pkl".format(region))

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    base = data_dir if data_dir is not None else DATA_DIR
    csv_path = os.path.join(base, REGION_FILE[region])
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            "[fetch_data] CSV not found: {}. "
            "Place {region}.csv files in the data/ directory.".format(
                csv_path, region=region
            )
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


# ---------------------------------------------------------------------------
# Relative return vectors (paper "Token Generation > Relative normalization")
# ---------------------------------------------------------------------------

def _compute_relative_returns(price_mat, w):
    """Compute (w-1)-dimensional relative return vectors.

    Paper formula:
        x_{i,t}^{[j]} = p_{i,(t-w+1+j)} / p_{i,(t-w+1)}
        x_{i,t} = [x^{[1]}, ..., x^{[w-1]}]^T in R^{w-1}

    Normalises price sequences by the window start price to remove
    scale differences and nonstationarity.

    Args:
        price_mat (ndarray): Shape (T, N) — closing price matrix.
        w (int): Price window size. Paper: w in {5, 10, 20}.

    Returns:
        ndarray: Shape (T, N, w-1). Rows with t < w-1 are initialised to 1.0.
    """
    T, N = price_mat.shape
    rel_ret = np.ones((T, N, w - 1), dtype=np.float32)

    for t in range(w - 1, T):
        base = price_mat[t - w + 1]
        safe_base = np.where(np.abs(base) < 1e-12, 1.0, base)
        for j in range(1, w):
            rel_ret[t, :, j - 1] = price_mat[t - w + 1 + j] / safe_base

    return rel_ret


# ---------------------------------------------------------------------------
# K-means tokenisation (paper "Token Generation > Centroid tokenization")
# ---------------------------------------------------------------------------

def _fit_kmeans(rel_ret, k, valid_start, train_end_idx):
    """Fit k-means on training-period relative return vectors.

    Paper: "Apply k-means clustering to the set X of x_{i,t} vectors to
    obtain k centroids v_1, ..., v_k." Only training data is used to
    prevent look-ahead bias.

    Paper Figure 2: k=200 is optimal on CSI300.

    Args:
        rel_ret (ndarray): Shape (T, N, w-1) — relative return vectors.
        k (int): Number of clusters.
        valid_start (int): First valid time index (= w-1).
        train_end_idx (int): Last index of the training period.

    Returns:
        sklearn.cluster.KMeans: Fitted model with .cluster_centers_ of
            shape (actual_k, w-1).
    """
    vecs = rel_ret[valid_start: train_end_idx + 1].reshape(-1, rel_ret.shape[-1])
    finite_mask = np.all(np.isfinite(vecs), axis=1)
    vecs = vecs[finite_mask]

    actual_k = min(k, len(vecs))
    km = KMeans(n_clusters=actual_k, random_state=42, n_init=10, max_iter=300)
    km.fit(vecs)
    return km


# ---------------------------------------------------------------------------
# SVD-based dynamic stock embedding (paper "Temporal stock embedding via SVD")
# ---------------------------------------------------------------------------

def _compute_svd_embeddings(daily_ret, d_embed=10, corr_window=60):
    """Compute temporal stock embeddings via SVD.

    Paper Eq. (1)-(2):
        Omega_t = Corr(R_t)              (Eq. 1)
        Omega_t = U_t Sigma_t U_t^T      (Eq. 2)
        E_t = U_t[:, :d]                 (top-d left singular vectors)

    A 60-trading-day (~3 month) rolling correlation matrix is decomposed
    via SVD to yield per-stock embeddings e_{i,t} that adapt dynamically
    to regime shifts and sector rotations.

    Args:
        daily_ret (ndarray): Shape (T, N) — daily return matrix.
        d_embed (int): Embedding dimension d. Default 10.
        corr_window (int): Rolling window length (trading days). Default 60.

    Returns:
        ndarray: Shape (T, N, d_embed). Rows with t < corr_window are zero.
    """
    T, N = daily_ret.shape
    emb = np.zeros((T, N, d_embed), dtype=np.float32)

    for t in range(corr_window, T):
        R = daily_ret[t - corr_window: t]
        std = R.std(axis=0)
        valid = std > 1e-8
        if valid.sum() < 2:
            continue

        R_v = R[:, valid]
        C = np.corrcoef(R_v.T)
        C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            U, _, _ = np.linalg.svd(C, full_matrices=False)
            d_use = min(d_embed, U.shape[1])
            cols = U[:, :d_use]
            if d_use < d_embed:
                cols = np.pad(cols, ((0, 0), (0, d_embed - d_use)))
            idx = np.where(valid)[0]
            emb[t, idx, :] = cols
        except np.linalg.LinAlgError:
            pass

    return emb


# ---------------------------------------------------------------------------
# Global context (paper "Global context via embedding similarities")
# ---------------------------------------------------------------------------

def _compute_global_context(stock_emb, token_emb, corr_window=60):
    """Compute global context via embedding-similarity weighted aggregation.

    Paper Eq. (3):
        g_{i,t} = sum_j alpha_{ij,t} * z_{j,t}
        alpha_{ij,t} = exp(-||e_{i,t} - e_{j,t}||^2) /
                        sum_k exp(-||e_{i,t} - e_{k,t}||^2)

    Stocks with similar embeddings contribute more to stock i's global
    context, thereby incorporating cross-stock price pattern information.

    Args:
        stock_emb (ndarray): Shape (T, N, d_embed) — SVD stock embeddings.
        token_emb (ndarray): Shape (T, N, F) — centroid token embeddings.
        corr_window (int): Minimum valid start time (= SVD window). Default 60.

    Returns:
        ndarray: Shape (T, N, F) — global context g_{i,t}.
            Rows with t < corr_window are zero.
    """
    T, N, _ = token_emb.shape
    gctx = np.zeros_like(token_emb)

    for t in range(corr_window, T):
        e = stock_emb[t]    # (N, d_embed)
        z = token_emb[t]    # (N, F)

        dist2 = np.sum((e[:, None, :] - e[None, :, :]) ** 2, axis=-1)  # (N, N)
        w = np.exp(-dist2)
        w /= w.sum(axis=1, keepdims=True) + 1e-12
        gctx[t] = w @ z     # (N, F)

    return gctx


# ---------------------------------------------------------------------------
# Level-1 cache: region base features (split-independent, computed once)
# ---------------------------------------------------------------------------

def cache_region_base(region, w=10, d_embed=10, corr_window=60,
                      cache_dir="cache", data_dir=None):
    """Compute and cache split-independent base features for a region.

    Level-1 cache contents:
      - price_mat:  (T, N) — closing price matrix
      - daily_ret:  (T, N) — daily returns
      - rel_ret:    (T, N, w-1) — relative return vectors
      - svd_emb:    (T, N, d_embed) — SVD stock embeddings

    These are independent of train/test split and are computed once per region.

    Args:
        region (str): Region code ("USA", "CHN", "JPN", "EUR").
        w (int): Price window size. Default 10.
        d_embed (int): SVD embedding dimension. Default 10.
        corr_window (int): Correlation rolling window. Default 60.
        cache_dir (str): Cache directory. Default "cache".
        data_dir (str or None): Override data directory.

    Returns:
        dict: Base feature dictionary with keys:
            "price_mat", "daily_ret", "rel_ret", "svd_emb",
            "dates", "symbols", "w", "d_embed", "corr_window".
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = "base_{}_w{}_d{}_c{}".format(region, w, d_embed, corr_window)
    cache_path = os.path.join(cache_dir, tag + ".pkl")

    if os.path.exists(cache_path):
        print("[Cache] Loading region base: {}".format(cache_path))
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[Cache] Building region base for {} ...".format(region))
    df = fetch_data(region, data_dir=data_dir, cache_dir=cache_dir)

    # Pivot to (date x symbol) closing price matrix (T, N)
    closes = df.pivot(index="date", columns="symbol", values="close").sort_index()
    closes = closes.ffill().bfill()

    symbols = closes.columns.tolist()
    dates = closes.index
    price_mat = closes.values.astype(np.float64)
    daily_ret = closes.pct_change().fillna(0.0).values.astype(np.float64)

    print("  computing relative returns ...")
    rel_ret = _compute_relative_returns(price_mat, w)

    print("  computing SVD embeddings ...")
    svd_emb = _compute_svd_embeddings(daily_ret, d_embed, corr_window)

    base = {
        "price_mat": price_mat,
        "daily_ret": daily_ret,
        "rel_ret": rel_ret,
        "svd_emb": svd_emb,
        "dates": dates,
        "symbols": symbols,
        "w": w,
        "d_embed": d_embed,
        "corr_window": corr_window,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(base, f)
    print("[Cache] Saved region base: {}".format(cache_path))
    return base


# ---------------------------------------------------------------------------
# Level-2: period-specific features (k-means + tokenisation + global context)
# ---------------------------------------------------------------------------

def build_period_features(base, k=200, train_end_date=None):
    """Build period-specific k-means tokens and global context from L1 base.

    Performs paper stages I1 (Token Generation) and I2 (Related Context):
      1. Fit k-means on training data -> centroids v_1, ..., v_k in R^{w-1}
      2. Assign each x_{i,t} to nearest centroid -> token ID phi(x_{i,t})
      3. Token embedding: z_{i,t} = v_{phi(x_{i,t})} in R^{w-1}
      4. Global context: g_{i,t} = sum_j alpha_{ij,t} * z_{j,t}  (Eq. 3)

    Args:
        base (dict): Output of cache_region_base().
        k (int): Number of centroids. Default 200 (paper Figure 2 optimum).
        train_end_date (str or None): Training period end date (YYYY-MM-DD).
            k-means is fit only up to this date to prevent look-ahead.

    Returns:
        dict: Period features with keys:
            "token_emb"   (T, N, F)   — centroid token embeddings z_{i,t}
            "global_ctx"  (T, N, F)   — global context g_{i,t}
            "token_ids"   (T, N)      — token IDs phi(x_{i,t}) in {0,..,k-1}
            "labels"      (T, N)      — movement labels y_{i,t} (1=up, 0=down)
            "daily_ret"   (T, N)      — daily returns
            "dates"       DatetimeIndex (T,)
            "symbols"     list[str] (N,)
            "n_clusters"  int — actual cluster count (<=k)
            "centroids"   (n_clusters, F) — centroid matrix
            "feat_dim"    int = w-1
            "min_valid_t" int — earliest valid time index for sequences
    """
    w = base["w"]
    rel_ret = base["rel_ret"]       # (T, N, w-1)
    dates = base["dates"]
    T = len(dates)
    valid_start = w - 1

    if train_end_date is not None:
        te = pd.to_datetime(train_end_date)
        train_end_idx = max(0, dates.searchsorted(te, side="right") - 1)
    else:
        train_end_idx = T - 1

    # k-means clustering (training data only)
    km = _fit_kmeans(rel_ret, k, valid_start, train_end_idx)
    centroids = km.cluster_centers_   # (actual_k, w-1)
    n_clusters = len(centroids)

    # Token assignment: phi(x_{i,t}) -> nearest centroid index
    flat = rel_ret.reshape(-1, w - 1)
    token_ids = km.predict(np.nan_to_num(flat, nan=1.0)).reshape(T, -1)
    token_emb = centroids[token_ids]  # (T, N, w-1)

    # Global context (Eq. 3)
    global_ctx = _compute_global_context(
        base["svd_emb"], token_emb, base["corr_window"]
    )

    # Movement labels: y_{i,t} = 1 if r_{i,t} > 0 else 0
    labels = (base["daily_ret"] > 0).astype(np.int64)
    min_valid_t = max(valid_start, base["corr_window"])

    return {
        "token_emb": token_emb.astype(np.float32),
        "global_ctx": global_ctx.astype(np.float32),
        "token_ids": token_ids.astype(np.int64),
        "labels": labels,
        "daily_ret": base["daily_ret"].astype(np.float32),
        "dates": dates,
        "symbols": base["symbols"],
        "n_clusters": n_clusters,
        "centroids": centroids.astype(np.float32),
        "feat_dim": w - 1,
        "min_valid_t": min_valid_t,
    }


# ---------------------------------------------------------------------------
# Legacy wrapper (main.py backward compatibility)
# ---------------------------------------------------------------------------

def compute_base_features(df, w=10, k=200, d_embed=10,
                          corr_window=60, train_end_date=None):
    """Single-call feature construction (for main.py; grid_search prefers
    the two-level caching via cache_region_base + build_period_features).

    Args:
        df (pd.DataFrame): Output of fetch_data() (raw price data).
        w (int): Price window size. Default 10.
        k (int): Number of centroids. Default 200.
        d_embed (int): SVD embedding dimension. Default 10.
        corr_window (int): Correlation window. Default 60.
        train_end_date (str or None): Training period end date.

    Returns:
        dict: Same output as build_period_features().
    """
    closes = df.pivot(index="date", columns="symbol", values="close").sort_index()
    closes = closes.ffill().bfill()

    symbols = closes.columns.tolist()
    dates = closes.index
    price_mat = closes.values.astype(np.float64)
    daily_ret = closes.pct_change().fillna(0.0).values.astype(np.float64)

    rel_ret = _compute_relative_returns(price_mat, w)
    svd_emb = _compute_svd_embeddings(daily_ret, d_embed, corr_window)

    base = {
        "price_mat": price_mat, "daily_ret": daily_ret,
        "rel_ret": rel_ret, "svd_emb": svd_emb,
        "dates": dates, "symbols": symbols,
        "w": w, "d_embed": d_embed, "corr_window": corr_window,
    }
    return build_period_features(base, k=k, train_end_date=train_end_date)


# ---------------------------------------------------------------------------
# Sequence construction (CRAFT model input format)
# ---------------------------------------------------------------------------

def build_sequences(features, seq_len):
    """Construct fixed-length input sequences and targets.

    For predicting date t+1:
        local  = token_emb[t-L+1 : t+1]       (L, N, F) local token sequence
        global = global_ctx[t-L+1 : t+1]       (L, N, F) global context
        token target  = token_ids[t-L+2 : t+2] (L, N)    1-step shifted
        movement target = labels[t+1]           (N,)      up/down label
        daily_return    = daily_ret[t+1]        (N,)      actual return

    Args:
        features (dict): Output of build_period_features().
        seq_len (int): Sequence length L. Paper: L in {5, 10, 20}.

    Returns:
        dict: Sequence data with keys:
            "local_seqs"       (S, L, N, F) — local token sequences
            "global_seqs"      (S, L, N, F) — global context sequences
            "token_targets"    (S, L, N)    — next-token targets
            "movement_targets" (S, N)       — movement labels y_{i,t+1}
            "daily_returns"    (S, N)       — actual daily returns (if available)
            "dates"            list[Timestamp] (S,) — prediction target dates
    """
    te = features["token_emb"]
    gc = features["global_ctx"]
    ti = features["token_ids"]
    lb = features["labels"]
    dr = features.get("daily_ret", None)
    dates = features["dates"]
    T = te.shape[0]
    start = max(features["min_valid_t"], seq_len - 1)

    locs, globs, ttgts, mtgts, drets, dts = [], [], [], [], [], []

    for t in range(start + seq_len - 1, T - 1):
        locs.append(te[t - seq_len + 1: t + 1])
        globs.append(gc[t - seq_len + 1: t + 1])
        ttgts.append(ti[t - seq_len + 2: t + 2])
        mtgts.append(lb[t + 1])
        if dr is not None:
            drets.append(dr[t + 1])
        dts.append(dates[t + 1])

    result = {
        "local_seqs": np.array(locs, dtype=np.float32),
        "global_seqs": np.array(globs, dtype=np.float32),
        "token_targets": np.array(ttgts, dtype=np.int64),
        "movement_targets": np.array(mtgts, dtype=np.int64),
        "dates": dts,
    }
    if drets:
        result["daily_returns"] = np.array(drets, dtype=np.float32)
    return result


def split_by_dates(seq_data, train_end, valid_end, test_start, test_end):
    """Split sequence data into train / valid / test by date boundaries.

    Paper Table 2 split scheme:
      - train : dates <= train_end
      - valid : train_end < dates <= valid_end
      - test  : test_start <= dates <= test_end

    Args:
        seq_data (dict): Output of build_sequences().
        train_end (str): Training period end date (YYYY-MM-DD).
        valid_end (str): Validation period end date.
        test_start (str): Test period start date.
        test_end (str): Test period end date.

    Returns:
        tuple: (train_d, valid_d, test_d) — each with the same structure
            as build_sequences() output, containing only the matching samples.
    """
    dates = pd.DatetimeIndex(seq_data["dates"])
    te = pd.to_datetime(train_end)
    ve = pd.to_datetime(valid_end)
    ts = pd.to_datetime(test_start)
    tend = pd.to_datetime(test_end)

    tr_mask = dates <= te
    va_mask = (dates > te) & (dates <= ve)
    tt_mask = (dates >= ts) & (dates <= tend)

    keys = ["local_seqs", "global_seqs", "token_targets", "movement_targets"]
    if "daily_returns" in seq_data:
        keys.append("daily_returns")

    def _slice(d, mask):
        out = {}
        for k in keys:
            out[k] = d[k][mask]
        out["dates"] = [d["dates"][i] for i in np.where(mask)[0]]
        return out

    return _slice(seq_data, tr_mask), _slice(seq_data, va_mask), _slice(seq_data, tt_mask)
