"""Train an action-conditioned latent flow-matching video world model."""

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

from world_model.data import PushCubeTransitionDataset
from world_model.flow import FlowVideoWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/flow_video_world_model"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=25, help="Use the first 25 state values (qpos, qvel, TCP) as proprioception.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def loss_and_metrics(model: FlowVideoWorldModel, batch: dict[str, torch.Tensor], state_dim: int) -> tuple[torch.Tensor, dict[str, float]]:
    state, next_state = batch["state"][:, :state_dim], batch["next_state"][:, :state_dim]
    current = model.encoder(batch["image"])
    next_latent = model.encoder(batch["next_image"])
    target = next_latent.detach()
    noise = torch.randn_like(target)
    time = torch.rand((len(target), 1), device=target.device)
    interpolation = (1 - time) * noise + time * target
    velocity_target = target - noise
    velocity = model.vector_field(interpolation, current, state, batch["action"], time)
    reconstruction = nn.functional.mse_loss(model.decoder(current), batch["image"])
    next_reconstruction = nn.functional.mse_loss(model.decoder(next_latent), batch["next_image"])
    flow = nn.functional.mse_loss(velocity, velocity_target)
    state_loss = nn.functional.mse_loss(model.state_head(current, state, batch["action"]), next_state)
    total = reconstruction + next_reconstruction + flow + state_loss
    return total, {"loss": total.item(), "recon_mse": reconstruction.item(), "next_recon_mse": next_reconstruction.item(), "flow_mse": flow.item(), "state_mse": state_loss.item()}


def evaluate(model: FlowVideoWorldModel, loader: DataLoader, state_dim: int, device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {}; batches = 0; model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            _, metrics = loss_and_metrics(model, batch, state_dim)
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return {f"val_{key}": value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PushCubeTransitionDataset(args.data, image_size=args.image_size)
    example = dataset[0]; full_state_dim, action_dim = example["state"].numel(), example["action"].numel(); dataset.close()
    if not 1 <= args.state_dim <= full_state_dim:
        raise ValueError(f"state-dim must be in [1, {full_state_dim}]")
    names = sorted({index.trajectory for index in dataset.indices}); random.Random(args.seed).shuffle(names)
    train_count = min(max(1, int(.9 * len(names))), len(names) - 1); train_names = set(names[:train_count])
    train_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory in train_names]
    val_indices = [i for i, index in enumerate(dataset.indices) if index.trajectory not in train_names]
    print(f"trajectory_split=train:{len(train_names)}, val:{len(names)-len(train_names)}; transitions=train:{len(train_indices)}, val:{len(val_indices)}")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_args)
    model = FlowVideoWorldModel(args.latent_dim, args.state_dim, action_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate); args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); totals: dict[str, float] = {}; batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True); loss, metrics = loss_and_metrics(model, batch, args.state_dim)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + value
            batches += 1
        row = {"epoch": epoch, **{key: value / batches for key, value in totals.items()}, **evaluate(model, val_loader, args.state_dim, device)}
        history.append(row); print(json.dumps(row, sort_keys=True))
        torch.save({"model": model.state_dict(), "args": vars(args), "metrics": row}, args.output / "last.pt")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
