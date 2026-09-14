"""Closed-loop CEM planning in a learned PushCube latent world model."""

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

from world_model.models import WorldModel


GOAL_XY = slice(25, 27)  # qpos(9), qvel(9), tcp_pose(7), goal_pos(3), obj_pose(7)
CUBE_XY = slice(28, 30)
TCP_POS = slice(18, 21)
GOAL_POS = slice(25, 28)
CUBE_POS = slice(28, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Replay HDF5 used to derive the demonstrator action prior.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--objective", choices=("terminal", "staged_dense"), default="staged_dense")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plan(
    model: WorldModel, image: torch.Tensor, state: torch.Tensor, action_mean: torch.Tensor,
    action_std: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor, args: argparse.Namespace,
) -> torch.Tensor:
    """Plan with either terminal distance or PushCube's staged dense objective."""
    action_dim = model.dynamics.net[0].in_features - model.dynamics.latent_dim - state.shape[-1]
    mean = action_mean.expand(args.horizon, action_dim).clone()
    std = action_std.expand(args.horizon, action_dim).clone()
    latent = model.encoder(image)
    for _ in range(args.iterations):
        actions = (mean + std * torch.randn(args.population, args.horizon, action_dim, device=state.device)).clamp(action_low, action_high)
        rollout_latent = latent.expand(args.population, -1)
        rollout_state = state.expand(args.population, -1)
        staged_cost = torch.zeros(args.population, device=state.device)
        for step in range(args.horizon):
            rollout_latent, rollout_state = model.dynamics(rollout_latent, rollout_state, actions[:, step])
            if args.objective == "staged_dense":
                # Match PushCubeEnv.compute_dense_reward: first reach the point
                # behind the cube, then receive placement reward.  This avoids
                # a terminal-only objective that provides no signal before contact.
                tcp_push = rollout_state[:, CUBE_POS] + torch.tensor(
                    [-0.025, 0.0, 0.0], device=state.device
                )
                reach_distance = torch.linalg.vector_norm(tcp_push - rollout_state[:, TCP_POS], dim=-1)
                goal_distance = torch.linalg.vector_norm(
                    rollout_state[:, CUBE_XY] - rollout_state[:, GOAL_XY], dim=-1
                )
                reaching_reward = 1 - torch.tanh(5 * reach_distance)
                placement_reward = 1 - torch.tanh(5 * goal_distance)
                reached = reach_distance < 0.01
                z_reward = 1 - torch.tanh(5 * (rollout_state[:, CUBE_POS.start + 2] - 0.02).abs())
                staged_reward = reaching_reward + reached * placement_reward * (1 + z_reward)
                staged_cost -= staged_reward
        distance = torch.linalg.vector_norm(rollout_state[:, CUBE_XY] - rollout_state[:, GOAL_XY], dim=-1)
        action_deviation = ((actions - action_mean) / action_std).square().mean(dim=(1, 2))
        if args.objective == "terminal":
            cost = distance
        else:
            cost = staged_cost / args.horizon + 0.002 * action_deviation
        elite_actions = actions[cost.topk(args.elites, largest=False).indices]
        mean, std = elite_actions.mean(dim=0), elite_actions.std(dim=0).clamp_min(0.05)
    return mean[0].clamp(action_low, action_high)


def action_prior(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as file:
        actions = np.concatenate([np.asarray(file[key]["actions"]) for key in file if key.startswith("traj_")])
    actions = torch.from_numpy(actions).float().to(device)
    return (
        actions.mean(dim=0), actions.std(dim=0).clamp_min(0.01),
        actions.amin(dim=0), actions.amax(dim=0),
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModel(settings["latent_dim"], 35, 8).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    action_mean, action_std, action_low, action_high = action_prior(args.data, device)
    print("action_prior_mean=", action_mean.tolist())
    env = gym.make(
        "PushCube-v1", num_envs=1, sim_backend="physx_cuda", obs_mode="rgb+state",
        control_mode="pd_joint_pos", render_mode="rgb_array", max_episode_steps=args.max_steps,
    )
    episodes: list[dict[str, float | bool | int]] = []
    with torch.no_grad():
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            success = False
            for step in range(args.max_steps):
                image = observation["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2).float().div(255)
                image = torch.nn.functional.interpolate(image, size=(settings["image_size"], settings["image_size"]), mode="bilinear", align_corners=False)
                action = plan(
                    model, image.to(device), observation["state"].to(device), action_mean, action_std,
                    action_low, action_high, args,
                )
                observation, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
                success = bool(info["success"][0])
                if success or bool(terminated[0]) or bool(truncated[0]):
                    break
            state = observation["state"][0]
            final_distance = float(torch.linalg.vector_norm(state[CUBE_XY] - state[GOAL_XY]))
            episodes.append({"episode": episode, "steps": step + 1, "success": success, "final_cube_goal_distance": final_distance})
            print(episodes[-1])
    env.close()
    report = {
        "planner": {key: getattr(args, key) for key in ("horizon", "population", "elites", "iterations", "objective")},
        "action_prior": {"mean": action_mean.tolist(), "std": action_std.tolist()},
        "episodes": episodes,
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "mean_final_cube_goal_distance": sum(item["final_cube_goal_distance"] for item in episodes) / len(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
