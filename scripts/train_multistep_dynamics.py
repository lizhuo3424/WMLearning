"""Fine-tune the visual latent world model with free multi-step rollouts."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model.data import PushCubeRolloutDataset
from world_model.models import WorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional one-step checkpoint for initialization")
    parser.add_argument("--output", type=Path, default=Path("outputs/multistep"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def rollout_loss(model: WorldModel, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    latent = model.encoder(batch["image"])
    reconstruction = nn.functional.mse_loss(model.decoder(latent), batch["image"])
    state = batch["state"]
    image_loss = state_loss = latent_loss = 0.0
    for step in range(batch["actions"].shape[1]):
        latent, state = model.dynamics(latent, state, batch["actions"][:, step])
        with torch.no_grad():
            target_latent = model.encoder(batch["next_images"][:, step])
        image_loss = image_loss + nn.functional.mse_loss(model.decoder(latent), batch["next_images"][:, step])
        state_loss = state_loss + nn.functional.mse_loss(state, batch["next_states"][:, step])
        latent_loss = latent_loss + nn.functional.mse_loss(latent, target_latent)
    horizon = batch["actions"].shape[1]
    image_loss, state_loss, latent_loss = image_loss / horizon, state_loss / horizon, latent_loss / horizon
    total = reconstruction + image_loss + state_loss + latent_loss
    return total, {"loss": total.item(), "recon_mse": reconstruction.item(), "rollout_image_mse": image_loss.item(), "rollout_state_mse": state_loss.item(), "rollout_latent_mse": latent_loss.item()}


def evaluate(model: WorldModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {}; batches = 0; model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            _, metrics = rollout_loss(model, batch)
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return {f"val_{key}": value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PushCubeRolloutDataset(args.data, args.horizon, args.image_size)
    example = dataset[0]; state_dim, action_dim = example["state"].numel(), example["actions"].shape[-1]
    dataset.close()
    trajectories = sorted({index.trajectory for index in dataset.indices}); random.Random(args.seed).shuffle(trajectories)
    train_count = min(max(1, int(.9 * len(trajectories))), len(trajectories) - 1)
    train_names = set(trajectories[:train_count])
    train_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory in train_names]
    val_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory not in train_names]
    print(f"horizon={args.horizon}; trajectory_split=train:{len(train_names)}, val:{len(trajectories)-len(train_names)}; rollouts=train:{len(train_indices)}, val:{len(val_indices)}")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_args)
    model = WorldModel(args.latent_dim, state_dim, action_dim).to(device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); print(f"initialized_from={args.checkpoint}")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate); args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); totals: dict[str, float] = {}; batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True); loss, metrics = rollout_loss(model, batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            batches += 1
        row = {"epoch": epoch, **{key: value / batches for key, value in totals.items()}, **evaluate(model, val_loader, device)}
        history.append(row); print(json.dumps(row, sort_keys=True))
        torch.save({"model": model.state_dict(), "args": vars(args), "metrics": row}, args.output / "last.pt")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
