"""Evaluate free latent rollouts and an action-conditioning ablation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model.models import WorldModel


IMAGE_KEY = "obs/sensor_data/base_camera/rgb"
STATE_KEY = "obs/state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-starts-per-trajectory", type=int, default=64)
    return parser.parse_args()


def to_image_tensor(images: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    return torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).to(device)


def validation_trajectories(file: h5py.File, seed: int) -> list[str]:
    trajectories = sorted(key for key in file if key.startswith("traj_"))
    random.Random(seed).shuffle(trajectories)
    train_count = min(max(1, int(0.9 * len(trajectories))), len(trajectories) - 1)
    return trajectories[train_count:]


def rollout_metrics(
    model: WorldModel,
    images: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    horizons: list[int],
    action_ablation: bool,
) -> tuple[dict[int, tuple[float, float]], int]:
    """Returns summed pixel/state MSE per horizon and number of starts."""
    max_horizon = max(horizons)
    latent = model.encoder(images[starts])
    predicted_state = states[starts]
    totals: dict[int, tuple[float, float]] = {}
    for step in range(1, max_horizon + 1):
        action = actions[starts + step - 1]
        if action_ablation:
            action = torch.zeros_like(action)
        latent, predicted_state = model.dynamics(latent, predicted_state, action)
        if step in horizons:
            target_image = images[starts + step]
            predicted_image = model.decoder(latent)
            pixel_mse = torch.nn.functional.mse_loss(predicted_image, target_image, reduction="none").mean(dim=(1, 2, 3))
            state_mse = torch.nn.functional.mse_loss(predicted_state, states[starts + step], reduction="none").mean(dim=1)
            totals[step] = (float(pixel_mse.sum()), float(state_mse.sum()))
    return totals, len(starts)


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",")})
    if min(horizons) < 1:
        raise ValueError("Horizons must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = {"conditioned": defaultdict(lambda: [0.0, 0.0]), "zero_action": defaultdict(lambda: [0.0, 0.0])}
    sample_count = 0

    with h5py.File(args.data, "r") as file:
        validation = validation_trajectories(file, args.seed)
        first = file[validation[0]]
        model = WorldModel(settings["latent_dim"], first[STATE_KEY].shape[-1], first["actions"].shape[-1]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            for trajectory in validation:
                group = file[trajectory]
                images = to_image_tensor(np.asarray(group[IMAGE_KEY]), settings["image_size"], device)
                states = torch.from_numpy(np.asarray(group[STATE_KEY])).float().to(device)
                actions = torch.from_numpy(np.asarray(group["actions"])).float().to(device)
                available = len(actions) - max(horizons) + 1
                if available <= 0:
                    continue
                starts = torch.arange(min(available, args.max_starts_per_trajectory), device=device)
                for label, ablation in (("conditioned", False), ("zero_action", True)):
                    metrics, count = rollout_metrics(model, images, states, actions, starts, horizons, ablation)
                    for horizon, (pixel_total, state_total) in metrics.items():
                        totals[label][horizon][0] += pixel_total
                        totals[label][horizon][1] += state_total
                sample_count += count

    results = {
        label: {
            str(horizon): {"pixel_mse": values[0] / sample_count, "state_mse": values[1] / sample_count}
            for horizon, values in sorted(per_horizon.items())
        }
        for label, per_horizon in totals.items()
    }
    report = {
        "checkpoint": str(args.checkpoint), "data": str(args.data), "validation_trajectories": len(validation),
        "rollout_starts": sample_count, "horizons": horizons, "metrics": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
