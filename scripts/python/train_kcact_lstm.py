"""Train LSTM model for Kcact prediction using patch-level time series.

Each patch_id forms a sequence of 8-day windows across the growing season.
The LSTM learns the temporal trajectory of Kcact from RS + weather features.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
}


class KcactLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        b, s, d = x.shape
        # Apply BatchNorm1d by reshaping: treat (b*s) as batch
        x_flat = x.reshape(-1, d)
        x_normed = self.input_bn(x_flat).reshape(b, s, d)
        lstm_out, _ = self.lstm(x_normed)
        lstm_out = self.ln(lstm_out)
        return self.head(lstm_out).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-table",
                        default=str(ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"))
    parser.add_argument("--train-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2024])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2023])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-seq-len", type=int, default=5,
                        help="Minimum sequence length (windows) per patch")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=["number", "bool"]).columns
            if col not in EXCLUDE_COLS and col != "year"]


def build_sequences(df: pd.DataFrame, feature_cols: list[str],
                    min_seq_len: int = 5):
    """Convert flat table to (patch_id, sequence) format."""
    df = df.sort_values(["patch_id", "date"]).copy()
    sequences, targets, patch_ids = [], [], []

    for pid, grp in df.groupby("patch_id"):
        grp_sorted = grp.sort_values("date")
        if len(grp_sorted) < min_seq_len:
            continue
        X_seq = grp_sorted[feature_cols].fillna(0.0).values.astype(np.float32)
        y_seq = grp_sorted["kcact"].values.astype(np.float32)
        sequences.append(X_seq)
        targets.append(y_seq)
        patch_ids.append(pid)

    return sequences, targets, patch_ids


def pad_sequences(sequences, targets, max_len=None):
    """Pad variable-length sequences to same length."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    input_dim = sequences[0].shape[1]

    X_padded = np.zeros((len(sequences), max_len, input_dim), dtype=np.float32)
    y_padded = np.zeros((len(targets), max_len), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=np.float32)

    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        n = len(seq)
        X_padded[i, :n, :] = seq
        y_padded[i, :n] = tgt
        mask[i, :n] = 1.0

    return X_padded, y_padded, mask


def masked_mse_loss(y_pred, y_true, mask):
    """MSE loss only on valid (non-padded) time steps."""
    diff = (y_pred - y_true) ** 2
    return (diff * mask).sum() / mask.sum()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for X_batch, y_batch, m_batch in loader:
        X_batch, y_batch, m_batch = X_batch.to(device), y_batch.to(device), m_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = masked_mse_loss(y_pred, y_batch, m_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    for X_batch, y_batch, m_batch in loader:
        X_batch, y_batch, m_batch = X_batch.to(device), y_batch.to(device), m_batch.to(device)
        y_pred = model(X_batch)
        # Gather only masked (valid) positions
        for i in range(len(y_batch)):
            valid_len = int(m_batch[i].sum().item())
            if valid_len > 0:
                y_true_all.extend(y_batch[i, :valid_len].cpu().numpy().tolist())
                y_pred_all.extend(y_pred[i, :valid_len].cpu().numpy().tolist())

    yt = np.array(y_true_all)
    yp = np.array(y_pred_all).clip(0.01, 2.0)
    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "n_samples": len(yt),
    }


def main():
    args = parse_args()
    device = torch.device(args.device or "cpu")  # default CPU: MPS has LSTM numerical issues
    print(f"Using device: {device}")

    df = pd.read_parquet(args.input_table)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    print(f"Features: {len(feature_cols)}")

    train_df = df[df["year"].isin(args.train_years)]
    test_df = df[df["year"].isin(args.test_years)]
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    train_seqs, train_tgts, _ = build_sequences(train_df, feature_cols, args.min_seq_len)
    test_seqs, test_tgts, _ = build_sequences(test_df, feature_cols, args.min_seq_len)
    print(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

    max_len = max(max(len(s) for s in train_seqs), max(len(s) for s in test_seqs))
    print(f"Max sequence length: {max_len}")

    X_train, y_train, m_train = pad_sequences(train_seqs, train_tgts, max_len)
    X_test, y_test, m_test = pad_sequences(test_seqs, test_tgts, max_len)

    # Standardize features using only training data (respect mask)
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    m_train_flat = m_train.reshape(-1)
    # Fit scaler only on valid (non-padded) training positions
    scaler = StandardScaler()
    valid_mask = m_train_flat > 0.5
    scaler.fit(X_train_flat[valid_mask])

    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)
    for i in range(seq_len):
        X_train_scaled[:, i, :] = scaler.transform(X_train[:, i, :])
        X_test_scaled[:, i, :] = scaler.transform(X_test[:, i, :])
    print(f"X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
    print(f"Feature means: {scaler.mean_[:5]}, stds: {scaler.scale_[:5]}")

    X_train, X_test = X_train_scaled, X_test_scaled

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(m_train))
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(m_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    model = KcactLSTM(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    best_rmse = float("inf")
    best_metrics = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_model(model, test_loader, device)
        scheduler.step(metrics["rmse"])

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | "
                  f"test R²={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")

        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n=== Best LSTM ({args.train_years}→{args.test_years}) ===")
    print(f"  R²={best_metrics['r2']:.4f}  RMSE={best_metrics['rmse']:.4f}  MAE={best_metrics['mae']:.4f}")

    # Save model
    out = Path(args.output_dir) / "models"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "kcact_lstm.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "max_seq_len": max_len,
        "args": vars(args),
    }, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
