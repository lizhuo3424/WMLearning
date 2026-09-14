"""Render an autoregressive sampled rollout: input | flow prediction | ground truth."""

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

from world_model.flow import FlowVideoWorldModel

IMAGE_KEY, STATE_KEY = "obs/sensor_data/base_camera/rgb", "obs/state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", default="traj_0")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_tensor(image: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    return torch.nn.functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).to(device)


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); settings = checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.data, "r") as file:
        group = file[args.trajectory]
        images = np.asarray(group[IMAGE_KEY]); states = torch.from_numpy(np.asarray(group[STATE_KEY])).float().to(device)
        actions = torch.from_numpy(np.asarray(group["actions"])).float().to(device)
    state_dim = settings.get("state_dim", states.shape[-1]); states = states[:, :state_dim]
    model = FlowVideoWorldModel(settings["latent_dim"], state_dim, actions.shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); frames: list[np.ndarray] = []
    with torch.no_grad():
        latent = model.encoder(image_tensor(images[0], settings["image_size"], device)); state = states[:1]
        for step in range(min(args.frames, len(actions))):
            input_frame = image_tensor(images[step], settings["image_size"], device)
            current_latent, current_state = latent, state
            latent = model.sample_next_latent(current_latent, current_state, actions[step:step + 1], steps=args.flow_steps)
            state = model.state_head(current_latent, current_state, actions[step:step + 1])
            prediction = model.decoder(latent)[0].permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
            current = input_frame[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
            target = image_tensor(images[step + 1], settings["image_size"], device)[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
            frames.append(np.concatenate((current, prediction, target), axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True); iio.imwrite(args.output, np.stack(frames), fps=args.fps)
    print(f"saved={args.output}; layout=input | flow_prediction | ground_truth; frames={len(frames)}")


if __name__ == "__main__":
    main()
