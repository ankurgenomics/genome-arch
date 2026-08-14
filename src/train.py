"""Training script for genome-arch's GenomicBenchmarks sanity-check stage.

Protocol matches genome-ft: a held-out validation split from train, the test
set scored exactly once on the best validation-MCC checkpoint, multiple seeds
with variance reported rather than a single number presented as final.

Usage:
    python -m src.train --task human_nontata_promoters --seeds 42 --epochs 1

For a fast local check without waiting on a full epoch over the real dataset:
    python -m src.train --task human_nontata_promoters --seeds 42 --epochs 1 --max_train 500
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.data import load_genomic_benchmark
from src.model import GenomeArchModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_seed(
    task: str,
    seed: int,
    epochs: int,
    block_size: int,
    embed_dim: int,
    batch_size: int,
    lr: float,
    max_train: int | None = None,
    device: str = "cpu",
) -> dict:
    set_seed(seed)
    (x_train_full, y_train_full), (x_test, y_test), seq_len = load_genomic_benchmark(task, block_size)

    if max_train is not None:
        x_train_full, y_train_full = x_train_full[:max_train], y_train_full[:max_train]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.15, random_state=seed, stratify=y_train_full
    )

    model = GenomeArchModel(seq_len=seq_len, embed_dim=embed_dim, block_size=block_size, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_mcc, best_state = -1.0, None
    n = x_train.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device)).argmax(dim=-1).cpu().numpy()
        val_mcc = matthews_corrcoef(y_val.numpy(), val_pred)
        if val_mcc > best_val_mcc:
            best_val_mcc, best_state = val_mcc, {k: v.clone() for k, v in model.state_dict().items()}

    # Test set scored exactly once, on the checkpoint with the best validation MCC.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test.to(device)).argmax(dim=-1).cpu().numpy()
    test_mcc = matthews_corrcoef(y_test.numpy(), test_pred)
    test_acc = float((test_pred == y_test.numpy()).mean())

    return {"seed": seed, "best_val_mcc": best_val_mcc, "test_mcc": test_mcc, "test_acc": test_acc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="human_nontata_promoters")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train", type=int, default=None, help="Truncate training set (for fast local runs)")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        r = run_one_seed(
            args.task, seed, args.epochs, args.block_size, args.embed_dim,
            args.batch_size, args.lr, max_train=args.max_train,
        )
        print(r)
        results.append(r)

    if len(results) > 1:
        mccs = [r["test_mcc"] for r in results]
        print(f"test_mcc mean={np.mean(mccs):.3f} std={np.std(mccs):.3f}")


if __name__ == "__main__":
    main()
