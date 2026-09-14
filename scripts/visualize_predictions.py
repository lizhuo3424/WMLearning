"""Render input / predicted-next-frame / ground-truth-next-frame comparison video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model.models import WorldModel


def image_tensor(image: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
    tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.to(device)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return image.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1).mul(255).byte().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", default="traj_0")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["args"]
    image_size, latent_dim = settings["image_size"], settings["latent_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.data, "r") as file:
        group = file[args.trajectory]
        states = np.asarray(group["obs/state"])
        actions = np.asarray(group["actions"])
        images = np.asarray(group["obs/sensor_data/base_camera/rgb"])
    model = WorldModel(latent_dim, states.shape[-1], actions.shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frames: list[np.ndarray] = []
    with torch.no_grad():
        for step in range(min(args.frames, len(actions))):
            image = image_tensor(images[step], image_size, device)
            state = torch.from_numpy(states[step]).float().unsqueeze(0).to(device)
            action = torch.from_numpy(actions[step]).float().unsqueeze(0).to(device)
            prediction = model(image, state, action)["predicted_image"]
            ground_truth = image_tensor(images[step + 1], image_size, device)
            frames.append(np.concatenate((to_uint8(image), to_uint8(prediction), to_uint8(ground_truth)), axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, np.stack(frames), fps=args.fps)
    print(f"saved={args.output}; layout=input | predicted_next | ground_truth_next; frames={len(frames)}")


if __name__ == "__main__":
    main()
