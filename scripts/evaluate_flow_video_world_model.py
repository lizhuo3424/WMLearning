"""Evaluate sampled free rollouts of the latent flow-matching video world model."""

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

from world_model.flow import FlowVideoWorldModel

IMAGE_KEY, STATE_KEY = "obs/sensor_data/base_camera/rgb", "obs/state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--max-starts-per-trajectory", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validation_trajectories(file: h5py.File, seed: int) -> list[str]:
    names = sorted(key for key in file if key.startswith("traj_"))
    random.Random(seed).shuffle(names)
    return names[min(max(1, int(.9 * len(names))), len(names) - 1):]


def image_tensor(images: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().div_(255)
    return torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).to(device)


def rollout(
    model: FlowVideoWorldModel, images: torch.Tensor, states: torch.Tensor, actions: torch.Tensor,
    starts: torch.Tensor, horizons: list[int], flow_steps: int, zero_action: bool,
) -> dict[int, tuple[float, float]]:
    latent = model.encoder(images[starts]); state = states[starts]
    totals: dict[int, tuple[float, float]] = {}
    for step in range(1, max(horizons) + 1):
        action = actions[starts + step - 1]
        if zero_action:
            action = torch.zeros_like(action)
        next_state = model.state_head(latent, state, action)
        next_latent = model.sample_next_latent(latent, state, action, steps=flow_steps)
        latent, state = next_latent, next_state
        if step in horizons:
            pixel_error = torch.nn.functional.mse_loss(model.decoder(latent), images[starts + step], reduction="none").mean(dim=(1, 2, 3))
            state_error = torch.nn.functional.mse_loss(state, states[starts + step], reduction="none").mean(dim=1)
            totals[step] = (float(pixel_error.sum()), float(state_error.sum()))
    return totals


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed)
    horizons = sorted({int(value) for value in args.horizons.split(",")})
    if min(horizons) < 1:
        raise ValueError("horizons must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["args"]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = {"conditioned": defaultdict(lambda: [0.0, 0.0]), "zero_action": defaultdict(lambda: [0.0, 0.0])}; starts_count = 0
    with h5py.File(args.data, "r") as file:
        validation = validation_trajectories(file, args.seed); first = file[validation[0]]
        state_dim = settings.get("state_dim", first[STATE_KEY].shape[-1])
        model = FlowVideoWorldModel(settings["latent_dim"], state_dim, first["actions"].shape[-1]).to(device)
        model.load_state_dict(checkpoint["model"]); model.eval()
        with torch.no_grad():
            for name in validation:
                group = file[name]; images = image_tensor(np.asarray(group[IMAGE_KEY]), settings["image_size"], device)
                states = torch.from_numpy(np.asarray(group[STATE_KEY])).float().to(device)[:, :state_dim]
                actions = torch.from_numpy(np.asarray(group["actions"])).float().to(device)
                available = len(actions) - max(horizons) + 1
                if available <= 0:
                    continue
                starts = torch.arange(min(available, args.max_starts_per_trajectory), device=device)
                for label, zero_action in (("conditioned", False), ("zero_action", True)):
                    errors = rollout(model, images, states, actions, starts, horizons, args.flow_steps, zero_action)
                    for horizon, (pixel, state) in errors.items():
                        totals[label][horizon][0] += pixel; totals[label][horizon][1] += state
                starts_count += len(starts)
    metrics = {label: {str(h): {"pixel_mse": values[0] / starts_count, "state_mse": values[1] / starts_count} for h, values in sorted(per_horizon.items())} for label, per_horizon in totals.items()}
    report = {"checkpoint": str(args.checkpoint), "data": str(args.data), "state_dim": state_dim, "validation_trajectories": len(validation), "rollout_starts": starts_count, "horizons": horizons, "flow_steps": args.flow_steps, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
