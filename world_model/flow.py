"""Action-conditioned latent flow-matching world model components."""

from __future__ import annotations

import math

import torch
from torch import nn

from world_model.models import VisualDecoder, VisualEncoder


class ConditionalVectorField(nn.Module):
    """Predicts a rectified-flow velocity for the next visual latent."""

    def __init__(self, latent_dim: int, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + state_dim + action_dim + 16, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def time_embedding(time: torch.Tensor, dim: int = 16) -> torch.Tensor:
        frequencies = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=time.device))
        phase = time * frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(
        self, noisy_next_latent: torch.Tensor, current_latent: torch.Tensor,
        state: torch.Tensor, action: torch.Tensor, time: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat((noisy_next_latent, current_latent, state, action, self.time_embedding(time)), dim=-1))


class StateTransitionHead(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, state_dim),
        )

    def forward(self, latent: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.net(torch.cat((latent, state, action), dim=-1))


class FlowVideoWorldModel(nn.Module):
    """CNN autoencoder + action-conditioned rectified flow in latent space."""

    def __init__(self, latent_dim: int, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
        self.vector_field = ConditionalVectorField(latent_dim, state_dim, action_dim)
        self.state_head = StateTransitionHead(latent_dim, state_dim, action_dim)
        self.latent_dim = latent_dim

    @torch.no_grad()
    def sample_next_latent(
        self, current_latent: torch.Tensor, state: torch.Tensor, action: torch.Tensor,
        steps: int = 8, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        sample = torch.randn(current_latent.shape, device=current_latent.device, generator=generator)
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(sample), 1), index * dt, device=sample.device)
            sample = sample + dt * self.vector_field(sample, current_latent, state, action, time)
        return sample
