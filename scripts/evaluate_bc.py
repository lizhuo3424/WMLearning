"""Evaluate the state-only behavior-cloning control baseline in real PushCube."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
os.environ.setdefault("VK_ICD_FILENAMES", str(PROJECT_ROOT / "configs" / "nvidia_icd_egl.json"))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from train_bc import BehaviorCloningPolicy, restore_actions

GOAL_XY, CUBE_XY = slice(25, 27), slice(28, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = BehaviorCloningPolicy(checkpoint["state_dim"], checkpoint["action_dim"]).to(device)
    policy.load_state_dict(checkpoint["model"]); policy.eval()
    state_mean, state_std = checkpoint["state_mean"].to(device), checkpoint["state_std"].to(device)
    action_mean, action_std = checkpoint["action_mean"].to(device), checkpoint["action_std"].to(device)
    env = gym.make("PushCube-v1", num_envs=1, sim_backend="physx_cuda", obs_mode="rgb+state", control_mode="pd_joint_pos", max_episode_steps=args.max_steps)
    episodes: list[dict[str, int | float | bool]] = []
    with torch.no_grad():
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode); success = False
            for step in range(args.max_steps):
                state_batch = observation["state"].to(device)
                target = policy((state_batch - state_mean) / state_std) * action_std + action_mean
                action = restore_actions(state_batch, target, checkpoint.get("action_representation", "raw"))
                observation, _, terminated, truncated, info = env.step(action)
                success = bool(info["success"][0])
                if success or bool(terminated[0]) or bool(truncated[0]): break
            state = observation["state"][0]
            record = {"episode": episode, "steps": step + 1, "success": success, "final_cube_goal_distance": float(torch.linalg.vector_norm(state[CUBE_XY] - state[GOAL_XY]))}
            episodes.append(record); print(record)
    env.close()
    report = {"policy": "state-only behavior cloning (privileged-state control baseline)", "episodes": episodes, "success_rate": sum(x["success"] for x in episodes) / len(episodes), "mean_final_cube_goal_distance": sum(x["final_cube_goal_distance"] for x in episodes) / len(episodes)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
