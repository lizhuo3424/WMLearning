"""Run a minimal ManiSkill PushCube GPU simulation and optionally save a video."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments


def describe_tree(value: Any, prefix: str = "obs") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            describe_tree(child, f"{prefix}.{key}")
        return
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    dtype = getattr(value, "dtype", type(value).__name__)
    print(f"{prefix}: shape={shape}, dtype={dtype}")


def to_frame(rendered: Any) -> np.ndarray:
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()
    rendered = np.asarray(rendered)
    if rendered.ndim == 4:
        rendered = rendered[0]
    if rendered.dtype != np.uint8:
        scale = 255.0 if rendered.max(initial=0) <= 1.0 else 1.0
        rendered = np.clip(rendered * scale, 0, 255).astype(np.uint8)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="physx_cuda")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vulkan-icd",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "nvidia_icd_egl.json",
        help="EGL-based NVIDIA Vulkan ICD JSON for headless cloud containers.",
    )
    args = parser.parse_args()

    if args.vulkan_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(args.vulkan_icd))
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

    env = gym.make(
        "PushCube-v1",
        num_envs=args.num_envs,
        sim_backend=args.backend,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=args.seed)
    print(f"VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES')}")
    print("Observation tree:")
    describe_tree(obs)
    print(f"action_space={env.action_space}")
    print(f"reset_info_keys={sorted(info.keys()) if isinstance(info, dict) else type(info)}")

    frames: list[np.ndarray] = []
    start = time.perf_counter()
    try:
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if args.video and step % 2 == 0:
                frames.append(to_frame(env.render()))
    finally:
        env.close()

    elapsed = time.perf_counter() - start
    transitions = args.steps * args.num_envs
    print(f"Completed {transitions} transitions in {elapsed:.2f}s ({transitions / elapsed:.1f} FPS)")

    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, frames, fps=30)
        print(f"Saved video: {args.video}")


if __name__ == "__main__":
    main()
