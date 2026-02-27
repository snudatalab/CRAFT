"""
engine.py — CRAFT model architecture, training loop, evaluation, and multi-GPU.

Paper: "Accurate Stock Movement Prediction via Centroid-based Randomness Smoothing"

Architecture overview (paper Figure 1):
  1. Centroid token sequence (local) + global context -> concatenated input
  2. Linear projection (2F -> d_model)
  3. Time-axis masked self-attention (decoder-style causal mask) — Eq. (4)-(5)
  4. Token prediction head (cross-entropy over k centroids) — Eq. (6)
  5. Stock-axis multi-head self-attention (last timestep) — Eq. (7)-(8)
  6. Movement prediction head (binary cross-entropy: up/down)

Total loss: L_total = L_time + lambda_stock * L_stock

Multi-GPU: nn.DataParallel (no process group or port binding needed).

Tensor dimension conventions:
    B = batch size
    L = sequence length (paper Table 1)
    N = number of stocks n (universe size)
    F = feature dimension = w-1 (relative return vector dim)
    k = number of centroids (paper: k=200 optimal)
    d = d_model (hidden dimension, paper: d in {64, 128, 256})
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class CRAFT(nn.Module):
    """CRAFT: Centroid-based Randomness Smoothing Approach for Stock
    Forecasting with Transformer Architecture.

    Implements the full pipeline from paper Figure 1:
      (1) Token Generation (handled in preprocess.py)
      (2) Time-axis Self-Attention -> next-token prediction (Eq. 4-6)
      (3) Stock-axis Multi-Head Self-Attention -> movement prediction (Eq. 7-8)

    Args:
        feat_dim (int): Feature dimension F = w-1.
        d_model (int): Transformer hidden dimension d. Default 64.
        n_heads (int): Number of attention heads. Default 2.
        n_layers (int): Number of Transformer encoder layers. Default 2.
        n_clusters (int): Number of k-means centroids k. Default 200.
        dropout (float): Dropout rate. Default 0.1.
    """

    def __init__(self, feat_dim, d_model=64, n_heads=2,
                 n_layers=2, n_clusters=200, dropout=0.1):
        super(CRAFT, self).__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model

        # Linear projection: [local || global] -> d_model
        # Paper: H = H_raw * W_proj, input dim = 2F = 2(w-1)
        self.proj = nn.Linear(2 * feat_dim, d_model)

        # Time-axis Transformer encoder (with causal mask)
        # Paper "Time-axis self-attention": Eq. (4)-(5)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.time_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Token prediction head: d -> k (centroid classification)
        # Paper Eq. 6: P(l_hat_tau = j) = softmax(W_out * h_tau^(1))_j
        self.token_head = nn.Linear(d_model, n_clusters)

        # Stock-axis multi-head self-attention (paper Eq. 7-8)
        self.stock_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.stock_ln = nn.LayerNorm(d_model)

        # Movement prediction head: d -> 2 (up/down binary classification)
        # Paper: o_i = W_F * h_i^(2)
        self.movement_head = nn.Linear(d_model, 2)

    def forward(self, local_seq, global_seq):
        """Full forward pass: local tokens + global context -> predictions.

        Args:
            local_seq (Tensor): Shape (B, L, N, F).
                Local centroid token sequence Z_{i,t}.
            global_seq (Tensor): Shape (B, L, N, F).
                Global context sequence G_{i,t} (Eq. 3).

        Returns:
            token_logits (Tensor): Shape (B, L, N, k).
                Per-(batch, timestep, stock) logits over k centroids.
            movement_logits (Tensor): Shape (B, N, 2).
                Per-(batch, stock) up/down logits.
        """
        B, L, N, F = local_seq.shape

        # (1) Concatenate local + global and project
        h = torch.cat([local_seq, global_seq], dim=-1)  # (B, L, N, 2F)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, 2 * F)
        h = self.proj(h)  # (B*N, L, d_model)

        # (2) Time-axis masked self-attention (Eq. 4-5)
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=h.device, dtype=h.dtype),
            diagonal=1,
        )
        h_time = self.time_encoder(h, mask=mask)  # (B*N, L, d_model)

        # (3) Token prediction head (Eq. 6)
        tok = self.token_head(h_time)  # (B*N, L, k)
        tok = tok.reshape(B, N, L, -1).permute(0, 2, 1, 3)  # (B, L, N, k)

        # (4) Stock-axis self-attention (Eq. 7-8) on last timestep
        h_last = h_time[:, -1, :].reshape(B, N, -1)  # (B, N, d_model)
        h_sa, _ = self.stock_attn(h_last, h_last, h_last)
        h_out = self.stock_ln(h_last + h_sa)  # (B, N, d_model)

        # (5) Movement prediction head
        mov = self.movement_head(h_out)  # (B, N, 2)
        return tok, mov


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _make_loader(split_data, batch_size, shuffle):
    """Convert split data dict to a PyTorch DataLoader.

    Args:
        split_data (dict): Output of preprocess.split_by_dates().
            Required keys: "local_seqs", "global_seqs", "token_targets",
            "movement_targets". Optional: "daily_returns".
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle (True for train, False otherwise).

    Returns:
        DataLoader: Yields (local, global, tok_target, mov_target [, daily_ret]).
    """
    tensors = [
        torch.from_numpy(split_data["local_seqs"]),
        torch.from_numpy(split_data["global_seqs"]),
        torch.from_numpy(split_data["token_targets"]),
        torch.from_numpy(split_data["movement_targets"]),
    ]
    if "daily_returns" in split_data:
        tensors.append(torch.from_numpy(split_data["daily_returns"]))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer,
                     ce_token, ce_move, lambda_stock, device):
    """Run one training epoch.

    Paper total loss: L_total = L_time + lambda_stock * L_stock
      - L_time  = CE over centroid token predictions (Eq. 6)
      - L_stock = CE over movement predictions

    Args:
        model (nn.Module): CRAFT model (may be DataParallel-wrapped).
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimiser instance.
        ce_token (nn.CrossEntropyLoss): Token prediction loss (L_time).
        ce_move (nn.CrossEntropyLoss): Movement prediction loss (L_stock).
        lambda_stock (float): L_stock weight lambda.
        device (str): Device string.

    Returns:
        float: Epoch-average total loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        local_b = batch[0].to(device)
        global_b = batch[1].to(device)
        tok_tgt = batch[2].to(device)
        mov_tgt = batch[3].to(device)

        tok_logits, mov_logits = model(local_b, global_b)

        B, L, N, K = tok_logits.shape
        loss_time = ce_token(tok_logits.reshape(-1, K), tok_tgt.reshape(-1))
        loss_stock = ce_move(mov_logits.reshape(-1, 2), mov_tgt.reshape(-1))
        loss = loss_time + lambda_stock * loss_stock

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_loss(model, loader, ce_token, ce_move, lambda_stock, device):
    """Compute L_total on a validation set (no gradients).

    Args:
        Same as _train_one_epoch().

    Returns:
        float: Validation-set average total loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        local_b = batch[0].to(device)
        global_b = batch[1].to(device)
        tok_tgt = batch[2].to(device)
        mov_tgt = batch[3].to(device)

        tok_logits, mov_logits = model(local_b, global_b)
        B, L, N, K = tok_logits.shape
        loss_time = ce_token(tok_logits.reshape(-1, K), tok_tgt.reshape(-1))
        loss_stock = ce_move(mov_logits.reshape(-1, 2), mov_tgt.reshape(-1))
        loss = loss_time + lambda_stock * loss_stock

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def backtest_stats(daily_rets):
    """Compute investment performance metrics from daily portfolio returns.

    Paper Evaluation Metrics:
      - ASR (Annualised Sharpe Ratio): (mu/sigma) * sqrt(252). Higher is better.
      - RMDD (Relative Maximum Drawdown): max peak-to-trough drop. Lower is better.
      - AVol (Annualised Volatility): sigma * sqrt(252).

    Args:
        daily_rets (array-like): Daily portfolio return sequence.

    Returns:
        tuple: (asr, rmdd, avol).
    """
    if len(daily_rets) == 0:
        return 0.0, 0.0, 0.0
    r = np.array(daily_rets, dtype=np.float64)
    mu = np.mean(r)
    sigma = np.std(r) + 1e-9

    avol = sigma * math.sqrt(252)
    asr = (mu / sigma) * math.sqrt(252)

    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-9)
    rmdd = float(np.abs(np.min(dd)))

    return asr, rmdd, avol


def compute_mcc(preds, targets):
    """Compute Matthews Correlation Coefficient (MCC).

    MCC = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))

    Args:
        preds (ndarray): Predicted labels (0 or 1), flattened.
        targets (ndarray): True labels (0 or 1), flattened.

    Returns:
        float: MCC in [-1, 1]. 1 = perfect, 0 = random.
    """
    tp = int(np.sum((preds == 1) & (targets == 1)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom < 1e-12:
        return 0.0
    return float(tp * tn - fp * fn) / denom


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Full evaluation: ACC, MCC, and (if daily_returns present) ASR, RMDD, AVol.

    Investment strategy (paper Investment Simulation, Tables 6-8):
      Equal-weight long/short portfolio based on predicted movement direction.
      - Predicted up  -> long position (+1)
      - Predicted down -> short position (-1)
      Daily rebalancing; portfolio return = mean(position * actual_return).

    Args:
        model (nn.Module): CRAFT model.
        loader (DataLoader): Test data loader.
        device (str): Device string.

    Returns:
        dict: Evaluation metrics with keys "ACC", "MCC", and optionally
            "ASR", "RMDD", "AVol".
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_daily_rets = []
    has_returns = False

    for batch in loader:
        local_b = batch[0].to(device)
        global_b = batch[1].to(device)
        mov_tgt = batch[3]

        _, mov_logits = model(local_b, global_b)
        preds = mov_logits.argmax(dim=-1).cpu()

        all_preds.append(preds)
        all_targets.append(mov_tgt)

        if len(batch) > 4:
            has_returns = True
            all_daily_rets.append(batch[4])

    preds_np = torch.cat(all_preds, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy()

    acc = float(np.mean(preds_np == targets_np))
    mcc = compute_mcc(preds_np.ravel(), targets_np.ravel())

    metrics = {"ACC": acc, "MCC": mcc}

    if has_returns and all_daily_rets:
        dr_np = torch.cat(all_daily_rets, dim=0).numpy()
        positions = np.where(preds_np == 1, 1.0, -1.0)
        portfolio_rets = np.mean(positions * dr_np, axis=1)

        asr, rmdd, avol = backtest_stats(portfolio_rets)
        metrics["ASR"] = asr
        metrics["RMDD"] = rmdd
        metrics["AVol"] = avol

    return metrics


# ---------------------------------------------------------------------------
# Multi-GPU wrapper
# ---------------------------------------------------------------------------

def _wrap_model(model):
    """Wrap model with nn.DataParallel if multiple GPUs are available.

    Args:
        model (nn.Module): CRAFT model instance.

    Returns:
        nn.Module: DataParallel-wrapped model (if GPU >= 2), else original.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def _unwrap_model(model):
    """Remove DataParallel wrapper and return the underlying model.

    Args:
        model (nn.Module): Possibly DataParallel-wrapped model.

    Returns:
        nn.Module: Original CRAFT model.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Public experiment runner
# ---------------------------------------------------------------------------

def run_experiment(train_data, valid_data, test_data,
                   n_clusters, feat_dim,
                   d_model=64, n_heads=2, n_layers=2,
                   lambda_stock=0.1, lr=1e-3, epochs=200,
                   batch_size=32, dropout=0.1, seed=42,
                   checkpoint_epochs=None, verbose=True):
    """Train CRAFT and evaluate at specified epoch checkpoints.

    Paper Experimental Settings:
      - Minimise L_total each epoch (Adam optimiser)
      - Save best model by validation loss
      - Report mean and std over 5 seeds (0-4)

    Args:
        train_data (dict): Training data (output of split_by_dates).
        valid_data (dict): Validation data.
        test_data (dict): Test data.
        n_clusters (int): Number of centroids k. Paper: 200.
        feat_dim (int): Feature dimension F = w-1.
        d_model (int): Hidden dimension d. Default 64.
        n_heads (int): Number of attention heads. Default 2.
        n_layers (int): Number of Transformer layers. Default 2.
        lambda_stock (float): L_stock weight lambda. Default 0.1.
        lr (float): Learning rate. Default 1e-3.
        epochs (int): Total training epochs. Default 200.
        batch_size (int): Batch size. Default 32.
        dropout (float): Dropout rate. Default 0.1.
        seed (int): Random seed. Default 42.
        checkpoint_epochs (list[int] or None): Epochs at which to evaluate.
            Default [20, 40, 60, 80, 100, 150, 200].
        verbose (bool): Whether to print results. Default True.

    Returns:
        dict: Mapping {epoch: metrics_dict}.
            Example: {200: {"ACC": 0.53, "MCC": 0.04, "ASR": 1.2, ...}}
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [20, 40, 60, 80, 100, 150, 200]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRAFT(
        feat_dim=feat_dim, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, n_clusters=n_clusters, dropout=dropout,
    ).to(device)
    model = _wrap_model(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_token = nn.CrossEntropyLoss()
    ce_move = nn.CrossEntropyLoss()

    train_loader = _make_loader(train_data, batch_size, shuffle=True)
    valid_loader = _make_loader(valid_data, batch_size, shuffle=False)
    test_loader = _make_loader(test_data, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    results = {}

    for epoch in range(1, epochs + 1):
        _train_one_epoch(
            model, train_loader, optimizer,
            ce_token, ce_move, lambda_stock, device
        )

        val_loss = _eval_loss(
            model, valid_loader,
            ce_token, ce_move, lambda_stock, device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(_unwrap_model(model).state_dict())

        if epoch in checkpoint_epochs:
            current_state = copy.deepcopy(_unwrap_model(model).state_dict())
            _unwrap_model(model).load_state_dict(best_state)
            metrics = evaluate_full(model, test_loader, device)
            _unwrap_model(model).load_state_dict(current_state)

            results[epoch] = metrics
            if verbose:
                m = metrics
                parts = ["epoch {:>4d}".format(epoch),
                         "val_loss {:.4f}".format(val_loss),
                         "ACC {:.4f}".format(m["ACC"]),
                         "MCC {:.4f}".format(m["MCC"])]
                if "ASR" in m:
                    parts.append("ASR {:.4f}".format(m["ASR"]))
                print("  " + " | ".join(parts))

    return results
