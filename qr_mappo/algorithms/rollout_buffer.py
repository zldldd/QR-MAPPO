"""Rollout buffer with GAE and soft-return estimation."""
from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_types: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.rollout_length = rollout_length
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.obs = np.zeros((rollout_length, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((rollout_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, n_agents), dtype=np.int64)
        self.log_probs = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.values = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.rewards = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.dones = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.agent_types = np.zeros((rollout_length, n_agents), dtype=np.int64)
        self.agent_ids = np.zeros((rollout_length, n_agents), dtype=np.int64)

        self.advantages = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.returns = np.zeros((rollout_length, n_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        agent_types: np.ndarray,
        agent_ids: np.ndarray,
    ):
        assert not self.full
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.agent_types[self.ptr] = agent_types
        self.agent_ids[self.ptr] = agent_ids
        self.ptr += 1
        if self.ptr == self.rollout_length:
            self.full = True

    def compute_returns_and_advantages(self, next_value: np.ndarray):
        """GAE-Lambda advantage estimation."""
        gae = np.zeros(self.n_agents, dtype=np.float32)
        next_value = next_value.astype(np.float32)
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_v = next_value
                next_done = np.zeros(self.n_agents, dtype=np.float32)
            else:
                next_v = self.values[t + 1]
                next_done = self.dones[t + 1]
            delta = self.rewards[t] + self.gamma * next_v * (1 - next_done) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Normalise advantages per agent.
        adv = self.advantages
        self.advantages = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-8)

    def get_batches(self, batch_size: int, n_epochs: int):
        """Yield random mini-batches flattened across time and agents."""
        T, N = self.rollout_length, self.n_agents
        batch_size = max(N, (batch_size // N) * N)
        indices = np.arange(T * N)
        data = {
            "obs": torch.from_numpy(self.obs.reshape(T * N, -1)).to(self.device),
            "states": torch.from_numpy(np.repeat(self.states, N, axis=0)).to(self.device),
            "actions": torch.from_numpy(self.actions.reshape(T * N)).to(self.device),
            "log_probs": torch.from_numpy(self.log_probs.reshape(T * N)).to(self.device),
            "returns": torch.from_numpy(self.returns.reshape(T * N)).to(self.device),
            "advantages": torch.from_numpy(self.advantages.reshape(T * N)).to(self.device),
            "agent_types": torch.from_numpy(self.agent_types.reshape(T * N)).to(self.device),
            "agent_ids": torch.from_numpy(self.agent_ids.reshape(T * N)).to(self.device),
        }
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                yield {k: v[batch_idx] for k, v in data.items()}

    def reset(self):
        self.ptr = 0
        self.full = False
