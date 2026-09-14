"""Train a state-based behavior-cloning baseline on PushCube demonstrations.

This is deliberately a *control baseline*, not the visual world model.  The
simulator state contains the object and goal poses, so this script establishes
whether the recorded demonstrations and action parameterization are learnable
before using the learned visual dynamics model for planning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/bc"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-representation", choices=("raw", "joint_delta"), default="raw",
        help="joint_delta predicts the first 7 PD joint-target offsets from qpos; gripper remains raw.",
    )
    return parser.parse_args()


def load_trajectory_split(path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load a whole-trajectory split, never splitting adjacent transitions."""
    with h5py.File(path, "r") as file:
        names = sorted(key for key in file if key.startswith("traj_"))
        if len(names) < 2:
            raise ValueError("Need at least two trajectories for a trajectory-level split")
        random.Random(seed).shuffle(names)
        train_count = min(max(1, int(.9 * len(names))), len(names) - 1)
        train_names, val_names = names[:train_count], names[train_count:]

        def arrays(selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.concatenate([np.asarray(file[name]["obs/state"])[:-1] for name in selected]).astype(np.float32),
                np.concatenate([np.asarray(file[name]["actions"]) for name in selected]).astype(np.float32),
            )

        train_states, train_actions = arrays(train_names)
        val_states, val_actions = arrays(val_names)
    return train_states, train_actions, val_states, val_actions, len(train_names), len(val_names)


def transform_actions(states: torch.Tensor, actions: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "raw":
        return actions
    transformed = actions.clone()
    transformed[..., :7] -= states[..., :7]
    return transformed


def restore_actions(states: torch.Tensor, targets: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "raw":
        return targets
    restored = targets.clone()
    restored[..., :7] += states[..., :7]
    return restored


def evaluate(
    model: BehaviorCloningPolicy, loader: DataLoader, state_mean: torch.Tensor, state_std: torch.Tensor,
    action_mean: torch.Tensor, action_std: torch.Tensor, representation: str, device: torch.device,
) -> tuple[float, float]:
    model.eval(); normalized_total = raw_total = 0.0; batches = 0
    with torch.no_grad():
        for state, action in loader:
            state, action = state.to(device), action.to(device)
            target = transform_actions(state, action, representation)
            prediction_norm = model((state - state_mean) / state_std)
            normalized_total += nn.functional.mse_loss(prediction_norm, (target - action_mean) / action_std).item()
            raw_total += nn.functional.mse_loss(restore_actions(state, prediction_norm * action_std + action_mean, representation), action).item()
            batches += 1
    return normalized_total / max(batches, 1), raw_total / max(batches, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_states, train_actions, val_states, val_actions, train_traj, val_traj = load_trajectory_split(args.data, args.seed)
    print(f"trajectory_split=train:{train_traj}, val:{val_traj}; transitions=train:{len(train_states)}, val:{len(val_states)}")
    state_mean = torch.from_numpy(train_states.mean(axis=0)).to(device)
    state_std = torch.from_numpy(train_states.std(axis=0).clip(1e-4)).to(device)
    train_targets = transform_actions(torch.from_numpy(train_states), torch.from_numpy(train_actions), args.action_representation).numpy()
    action_mean = torch.from_numpy(train_targets.mean(axis=0)).to(device)
    action_std = torch.from_numpy(train_targets.std(axis=0).clip(1e-4)).to(device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_states), torch.from_numpy(train_actions)), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_states), torch.from_numpy(val_actions)), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model = BehaviorCloningPolicy(train_states.shape[1], train_actions.shape[1]).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; batches = 0
        for state, action in train_loader:
            state, action = state.to(device), action.to(device)
            prediction = model((state - state_mean) / state_std)
            target = transform_actions(state, action, args.action_representation)
            loss = nn.functional.mse_loss(prediction, (target - action_mean) / action_std)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            total += loss.item(); batches += 1
        val_norm_mse, val_raw_mse = evaluate(model, val_loader, state_mean, state_std, action_mean, action_std, args.action_representation, device)
        row = {"epoch": epoch, "train_normalized_action_mse": total / batches, "val_normalized_action_mse": val_norm_mse, "val_raw_action_mse": val_raw_mse}
        history.append(row); print(json.dumps(row, sort_keys=True))
        torch.save({"model": model.state_dict(), "state_mean": state_mean.cpu(), "state_std": state_std.cpu(), "action_mean": action_mean.cpu(), "action_std": action_std.cpu(), "action_representation": args.action_representation, "state_dim": train_states.shape[1], "action_dim": train_actions.shape[1], "args": vars(args), "metrics": row}, args.output / "last.pt")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
