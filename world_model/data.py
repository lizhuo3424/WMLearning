from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TransitionIndex:
    trajectory: str
    step: int


class PushCubeTransitionDataset(Dataset):
    """One-step RGB/state/action transitions from a ManiSkill replay HDF5 file.

    Each item follows the supervised world-model convention:
    ``(o_t, s_t, a_t) -> (o_{t+1}, s_{t+1})``.
    The HDF5 handle is opened lazily so DataLoader workers do not share it.
    """

    image_key = "obs/sensor_data/base_camera/rgb"
    state_key = "obs/state"

    def __init__(self, path: str | Path, image_size: int = 64) -> None:
        self.path = str(path)
        self.image_size = image_size
        self._file: h5py.File | None = None
        self.indices: list[TransitionIndex] = []

        with h5py.File(self.path, "r") as file:
            for trajectory in sorted(key for key in file if key.startswith("traj_")):
                actions = file[f"{trajectory}/actions"]
                observations = file[f"{trajectory}/{self.image_key}"]
                if len(observations) != len(actions) + 1:
                    raise ValueError(
                        f"{trajectory}: expected T+1 observations for T actions, "
                        f"got {len(observations)} and {len(actions)}"
                    )
                self.indices.extend(TransitionIndex(trajectory, step) for step in range(len(actions)))

        if not self.indices:
            raise ValueError(f"No transitions found in {self.path}")

    def __len__(self) -> int:
        return len(self.indices)

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    @staticmethod
    def _image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        if tensor.shape[-1] != image_size or tensor.shape[-2] != image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        return tensor

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        group = self._handle()[index.trajectory]
        step = index.step
        images = group[self.image_key]
        states = group[self.state_key]
        return {
            "image": self._image_to_tensor(images[step], self.image_size),
            "next_image": self._image_to_tensor(images[step + 1], self.image_size),
            "state": torch.from_numpy(states[step].astype(np.float32)),
            "next_state": torch.from_numpy(states[step + 1].astype(np.float32)),
            "action": torch.from_numpy(group["actions"][step].astype(np.float32)),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


@dataclass(frozen=True)
class RolloutIndex:
    trajectory: str
    step: int


class PushCubeRolloutDataset(PushCubeTransitionDataset):
    """Fixed-horizon sequences for free latent rollout training.

    An item begins at ``t`` and contains H actions plus targets through
    ``t + H``.  No sequence crosses a demonstration boundary.
    """

    def __init__(self, path: str | Path, horizon: int, image_size: int = 64) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.path = str(path)
        self.image_size = image_size
        self.horizon = horizon
        self._file: h5py.File | None = None
        self.indices: list[RolloutIndex] = []
        with h5py.File(self.path, "r") as file:
            for trajectory in sorted(key for key in file if key.startswith("traj_")):
                action_count = len(file[f"{trajectory}/actions"])
                self.indices.extend(
                    RolloutIndex(trajectory, step) for step in range(max(0, action_count - horizon + 1))
                )
        if not self.indices:
            raise ValueError(f"No {horizon}-step rollouts found in {self.path}")

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        group = self._handle()[index.trajectory]
        start, stop = index.step, index.step + self.horizon
        images = group[self.image_key]
        states = group[self.state_key]
        return {
            "image": self._image_to_tensor(images[start], self.image_size),
            "state": torch.from_numpy(states[start].astype(np.float32)),
            "actions": torch.from_numpy(group["actions"][start:stop].astype(np.float32)),
            "next_images": torch.stack([
                self._image_to_tensor(images[step], self.image_size) for step in range(start + 1, stop + 1)
            ]),
            "next_states": torch.from_numpy(states[start + 1:stop + 1].astype(np.float32)),
        }
