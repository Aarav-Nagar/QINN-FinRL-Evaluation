"""Classical MPS regressor used by the Agentic Trading Lab adapter.

This is a small classical tensor-network model. It does not use quantum
hardware. The local feature map and contraction match the MPS formulation in
``run_experiment.py`` while allowing an ATL-specific input distribution.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MPSRegressor(nn.Module):
    """Matrix-product-state regressor with a quantum-style local feature map."""

    def __init__(
        self, feature_count: int = 13, bond_dimension: int = 4, *, seed: int = 2026
    ):
        super().__init__()
        if feature_count < 2:
            raise ValueError("feature_count must be at least 2")
        if bond_dimension < 1:
            raise ValueError("bond_dimension must be positive")
        ranks = [1] + [bond_dimension] * (feature_count - 1) + [1]
        generator = torch.Generator().manual_seed(seed)
        cores = []
        for left_rank, right_rank in zip(ranks[:-1], ranks[1:]):
            scale = 1.0 / math.sqrt(2 * left_rank)
            core = torch.randn(
                left_rank,
                2,
                right_rank,
                generator=generator,
                dtype=torch.float32,
            ) * scale
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @staticmethod
    def local_feature_map(inputs: torch.Tensor) -> torch.Tensor:
        clipped = inputs.clamp(-3.0, 3.0) / 3.0
        angles = (math.pi / 4.0) * (clipped + 1.0)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.local_feature_map(inputs)
        state = torch.ones(
            (inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device
        )
        for site, core in enumerate(self.cores):
            state = torch.einsum("br,ris,bi->bs", state, core, local[:, site, :])
        return state.squeeze(-1) + self.bias


class MatchedANNRegressor(nn.Module):
    """A 369-parameter neural baseline matched to the 13-input MPS."""

    def __init__(self, feature_count: int = 13):
        super().__init__()
        if feature_count != 13:
            raise ValueError("The matched architecture is defined for 13 features")
        self.network = nn.Sequential(
            nn.Linear(13, 20),
            nn.Tanh(),
            nn.Linear(20, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class ResidualMPSRegressor(nn.Module):
    """Bond-5 MPS with a learnable linear residual path (586 parameters)."""

    def __init__(self, feature_count: int = 13, *, seed: int = 2026):
        super().__init__()
        if feature_count != 13:
            raise ValueError("The residual architecture is defined for 13 features")
        self.mps = MPSRegressor(feature_count, bond_dimension=5, seed=seed)
        self.linear_weight = nn.Parameter(torch.zeros(feature_count))
        self.mps_scale = nn.Parameter(torch.ones(()))
        self.linear_scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mps_scale * self.mps(inputs) + self.linear_scale * (
            inputs @ self.linear_weight
        )


class MatchedANNV3Regressor(nn.Module):
    """A one-hidden-layer ANN exactly matched to v3's 586 parameters."""

    def __init__(self, feature_count: int = 13):
        super().__init__()
        if feature_count != 13:
            raise ValueError("The matched architecture is defined for 13 features")
        self.network = nn.Sequential(
            nn.Linear(13, 39),
            nn.Tanh(),
            nn.Linear(39, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
