from __future__ import annotations

import torch
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, latent_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(latent_dim, 256 * 4 * 4), nn.SiLU())
        self.net = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(self.project(latent))


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim + state_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, latent: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.net(torch.cat((latent, state, action), dim=-1))
        delta_latent, delta_state = delta.split((self.latent_dim, state.shape[-1]), dim=-1)
        return latent + delta_latent, state + delta_state


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
        self.dynamics = LatentDynamics(latent_dim, state_dim, action_dim)

    def forward(self, image: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(image)
        predicted_latent, predicted_state = self.dynamics(latent, state, action)
        return {
            "latent": latent,
            "reconstruction": self.decoder(latent),
            "predicted_latent": predicted_latent,
            "predicted_state": predicted_state,
            "predicted_image": self.decoder(predicted_latent),
        }
