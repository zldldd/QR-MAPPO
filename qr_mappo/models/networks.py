"""Neural-network building blocks for QR-MAPPO."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * layers + [output_dim]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(nn.LayerNorm(dims[i + 1]))
                blocks.append(nn.ReLU())
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentEmbedding(nn.Module):
    """Concatenated agent-type and agent-id embedding."""

    def __init__(self, n_types: int, n_ids: int, type_dim: int, id_dim: int):
        super().__init__()
        self.type_embed = nn.Embedding(n_types, type_dim)
        self.id_embed = nn.Embedding(n_ids, id_dim)

    def forward(self, agent_type: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.type_embed(agent_type), self.id_embed(agent_id)], dim=-1)


class RationalityEncoder(nn.Module):
    """Estimates adversary rationality rho and produces a conditioning vector."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, embed_dim: int = 32):
        super().__init__()
        self.encoder = MLP(state_dim, hidden_dim, hidden_dim, layers=2)
        self.rho_head = nn.Linear(hidden_dim, 1)
        self.embed_head = nn.Linear(hidden_dim, embed_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.encoder(state))
        rho = F.softplus(self.rho_head(h)) + 0.1
        embed = F.relu(self.embed_head(h))
        return rho, embed


class AttentionCommunication(nn.Module):
    """Multi-head gated attention for inter-agent message passing."""

    def __init__(self, embed_dim: int, n_heads: int = 4, comm_dim: int = 64):
        super().__init__()
        assert comm_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = comm_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, comm_dim)
        self.k_proj = nn.Linear(embed_dim, comm_dim)
        self.v_proj = nn.Linear(embed_dim, comm_dim)
        self.out_proj = nn.Linear(comm_dim, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim)
        B, N, E = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        msg = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        msg = self.out_proj(msg)
        gate = self.gate(torch.cat([x, msg], dim=-1))
        return x + gate * msg


class ParameterSpacePerturbation:
    """In-place parameter-space perturbation (PSP) helper."""

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def backup(self, model: nn.Module) -> dict:
        return {name: p.data.clone() for name, p in model.named_parameters()}

    def perturb(self, model: nn.Module, rng: Optional[torch.Generator] = None):
        for p in model.parameters():
            std = self.epsilon * p.data.abs().clamp_min(1e-8)
            noise = torch.randn_like(p.data) if rng is None else torch.randn_like(p.data, generator=rng)
            p.data.add_(std * noise)

    def restore(self, model: nn.Module, backup: dict):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])


class QREActor(nn.Module):
    """Quantal-Response Equilibrium actor with optional communication and
    adversary-rationality conditioning.
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_types: int,
        action_dim: int,
        hidden_dim: int = 256,
        agent_type_dim: int = 15,
        agent_id_dim: int = 16,
        comm_dim: int = 64,
        n_comm_heads: int = 4,
        use_comm: bool = True,
        use_rho: bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.use_comm = use_comm
        self.use_rho = use_rho

        self.agent_embed = AgentEmbedding(n_types, n_agents, agent_type_dim, agent_id_dim)
        embed_in = obs_dim + agent_type_dim + agent_id_dim
        self.encoder = MLP(embed_in, hidden_dim, hidden_dim, layers=2)

        if use_comm:
            self.comm = AttentionCommunication(hidden_dim, n_comm_heads, comm_dim)

        if use_rho:
            self.rho_fusion = nn.Linear(hidden_dim + 32, hidden_dim)

        # Per-action-head logits and a learnable temperature.
        self.logits_head = nn.Linear(hidden_dim, action_dim)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        obs: torch.Tensor,
        agent_type: torch.Tensor,
        agent_id: torch.Tensor,
        rho_embed: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        # obs: (B, N, obs_dim)
        B, N, _ = obs.shape
        type_emb = self.agent_embed(agent_type, agent_id)  # (B, N, type_dim+id_dim)
        x = torch.cat([obs, type_emb], dim=-1)
        h = self.encoder(x)  # (B, N, hidden_dim)

        if self.use_comm:
            h = self.comm(h)

        if self.use_rho and rho_embed is not None:
            if rho_embed.dim() == 2:
                rho_embed = rho_embed.unsqueeze(1).expand(-1, N, -1)
            h = F.relu(self.rho_fusion(torch.cat([h, rho_embed], dim=-1)))

        logits = self.logits_head(h)  # (B, N, action_dim)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        temperature = torch.exp(self.log_temp).clamp_min(1e-4)
        dist = Categorical(logits=logits / temperature)
        return dist, temperature


class DualSoftCritic(nn.Module):
    """Dual value-network critic for PPO target-value clipping."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.v1 = MLP(state_dim, hidden_dim, 1, layers=2)
        self.v2 = MLP(state_dim, hidden_dim, 1, layers=2)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.v1(state).squeeze(-1), self.v2(state).squeeze(-1)
