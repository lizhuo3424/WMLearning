"""State-retrieval imitation baseline for diagnosing demonstration coverage.

At each environment step, retrieve nearest expert states by robot qpos, TCP,
cube and goal pose, then execute their (possibly distance-weighted) actions.
This is a non-parametric privileged-state baseline, not a visual policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
os.environ.setdefault("VK_ICD_FILENAMES", str(PROJECT_ROOT / "configs" / "nvidia_icd_egl.json"))

import gymnasium as gym
import mani_skill.envs  # noqa: F401

FEATURES = list(range(9)) + list(range(18, 21)) + list(range(25, 31))
GOAL_XY, CUBE_XY = slice(25, 27), slice(28, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1, help="Expert actions executed before the next retrieval.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_bank(path: Path, chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as file:
        names = sorted(key for key in file if key.startswith("traj_"))
        trajectory_states: list[np.ndarray] = []
        trajectory_chunks: list[np.ndarray] = []
        for name in names:
            states = np.asarray(file[name]["obs/state"])[:-1]
            actions = np.asarray(file[name]["actions"])
            chunks = np.empty((len(actions), chunk_size, actions.shape[-1]), dtype=np.float32)
            for step in range(len(actions)):
                tail = actions[step : step + chunk_size]
                chunks[step, :len(tail)] = tail
                if len(tail) < chunk_size:
                    chunks[step, len(tail):] = tail[-1]
            trajectory_states.append(states)
            trajectory_chunks.append(chunks)
        states = np.concatenate(trajectory_states).astype(np.float32)
        action_chunks = np.concatenate(trajectory_chunks)
    features = torch.from_numpy(states[:, FEATURES]).to(device)
    mean, std = features.mean(0), features.std(0).clamp_min(1e-4)
    return (features - mean) / std, torch.from_numpy(action_chunks).to(device), mean, std


def main() -> None:
    args = parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    bank, action_chunks, mean, std = load_bank(args.data, args.chunk_size, device)
    env = gym.make("PushCube-v1", num_envs=1, sim_backend="physx_cuda", obs_mode="rgb+state", control_mode="pd_joint_pos", max_episode_steps=args.max_steps)
    episodes: list[dict[str, int | float | bool]] = []
    with torch.no_grad():
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode); success = False
            chunk: torch.Tensor | None = None; chunk_offset = 0
            for step in range(args.max_steps):
                if chunk is None or chunk_offset == args.chunk_size:
                    query = (observation["state"][:, FEATURES].to(device) - mean) / std
                    distance = torch.cdist(query, bank).squeeze(0)
                    closest = distance.topk(args.neighbors, largest=False)
                    if args.neighbors == 1:
                        chunk = action_chunks[closest.indices[0]]
                    else:
                        weights = torch.softmax(-closest.values / 0.25, dim=0)
                        chunk = (weights[:, None, None] * action_chunks[closest.indices]).sum(0)
                    chunk_offset = 0
                action = chunk[chunk_offset].unsqueeze(0)
                chunk_offset += 1
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"][0])
                if success or bool(terminated[0]) or bool(truncated[0]): break
            state = observation["state"][0]
            record = {"episode": episode, "steps": step + 1, "success": success, "final_cube_goal_distance": float(torch.linalg.vector_norm(state[CUBE_XY] - state[GOAL_XY]))}
            episodes.append(record); print(record)
    env.close()
    report = {"policy": "privileged-state kNN action-chunk retrieval", "features": FEATURES, "neighbors": args.neighbors, "chunk_size": args.chunk_size, "bank_transitions": len(bank), "episodes": episodes, "success_rate": sum(x["success"] for x in episodes) / len(episodes), "mean_final_cube_goal_distance": sum(x["final_cube_goal_distance"] for x in episodes) / len(episodes)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
