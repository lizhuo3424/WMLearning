"""Train a first action-conditioned visual latent-dynamics baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Permit `python scripts/train_dynamics.py` from a fresh checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model.data import PushCubeTransitionDataset
from world_model.models import WorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def loss_and_metrics(model: WorldModel, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["image"], batch["state"], batch["action"])
    with torch.no_grad():
        next_latent = model.encoder(batch["next_image"])
    reconstruction = nn.functional.mse_loss(output["reconstruction"], batch["image"])
    next_reconstruction = nn.functional.mse_loss(output["predicted_image"], batch["next_image"])
    latent_dynamics = nn.functional.mse_loss(output["predicted_latent"], next_latent)
    state_dynamics = nn.functional.mse_loss(output["predicted_state"], batch["next_state"])
    total = reconstruction + next_reconstruction + latent_dynamics + state_dynamics
    return total, {
        "loss": total.item(), "recon_mse": reconstruction.item(),
        "next_image_mse": next_reconstruction.item(), "latent_mse": latent_dynamics.item(),
        "state_mse": state_dynamics.item(),
    }


def evaluate(model: WorldModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            _, metrics = loss_and_metrics(model, batch)
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            count += 1
    return {f"val_{key}": value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PushCubeTransitionDataset(args.data, image_size=args.image_size)
    example = dataset[0]
    state_dim, action_dim = example["state"].numel(), example["action"].numel()
    dataset.close()  # Do not fork a live h5py handle into DataLoader workers.
    # Split whole demonstrations, never random transitions: the latter leaks
    # near-identical adjacent frames from the same trajectory into validation.
    trajectories = sorted({index.trajectory for index in dataset.indices})
    if len(trajectories) < 2:
        raise ValueError("Need at least two trajectories for a trajectory-level train/validation split")
    random.Random(args.seed).shuffle(trajectories)
    train_trajectory_count = min(max(1, int(0.9 * len(trajectories))), len(trajectories) - 1)
    train_trajectories = set(trajectories[:train_trajectory_count])
    train_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory in train_trajectories]
    val_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory not in train_trajectories]
    train_set, val_set = Subset(dataset, train_indices), Subset(dataset, val_indices)
    print(f"trajectory_split=train:{len(train_trajectories)}, val:{len(trajectories) - len(train_trajectories)}; "
          f"transitions=train:{len(train_indices)}, val:{len(val_indices)}")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = WorldModel(args.latent_dim, state_dim, action_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); totals: dict[str, float] = {}; batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_and_metrics(model, batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            batches += 1
        row = {"epoch": epoch, **{key: value / batches for key, value in totals.items()}, **evaluate(model, val_loader, device)}
        history.append(row); print(json.dumps(row, sort_keys=True))
        torch.save({"model": model.state_dict(), "args": vars(args), "metrics": row}, args.output / "last.pt")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
