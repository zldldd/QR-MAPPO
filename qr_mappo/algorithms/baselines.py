"""Simple deterministic/heuristic baselines for comparison."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class RandomPolicy:
    def __init__(self, env: Any):
        self.env = env
        self.agents = env.get_agent_ids()

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        return {a: self.env.rng.integers(0, self.env.action_space_size) for a in self.agents}


class HeuristicPolicy:
    """Hand-engineered policy: radars detect, jammers jam, shooters engage."""

    def __init__(self, env: Any):
        self.env = env
        self.agents = env.get_agent_ids()

    def _role(self, agent_id: str) -> str:
        idx = int(agent_id[1:])
        if idx <= 4:
            return "radar"
        if idx == 5:
            return "jammer"
        if idx <= 10:
            return "missile"
        if idx <= 13:
            return "gun"
        return "command"

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        alive_targets = [t for t in self.env.targets if t.alive]

        def nearest(agent_pos: np.ndarray, targets: List[Any]) -> int:
            if not targets:
                return 0
            return min(targets, key=lambda t: np.linalg.norm(agent_pos - t.position)).id

        for a in self.env.agents:
            role = self._role(a.id)
            if role == "command":
                actions[a.id] = 0
                continue
            if not alive_targets:
                actions[a.id] = 0
                continue
            t_id = nearest(a.position, alive_targets)
            actions[a.id] = t_id + 1
        return actions


def rollout_policy(env: Any, policy: Any, n_episodes: int = 20) -> Dict[str, float]:
    returns = []
    kills = []
    leaks = []
    for ep in range(n_episodes):
        obs, _, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        while not done:
            actions = policy.select_actions(obs)
            obs, _, rewards, terms, truncs, infos = env.step(actions)
            ep_return += float(np.mean([rewards[a] for a in env.get_agent_ids()]))
            done = any(terms.values()) or any(truncs.values())
        returns.append(ep_return)
        info = infos[env.get_agent_ids()[0]]
        kills.append(info["killed"])
        leaks.append(info["leaked"])
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_killed": float(np.mean(kills)),
        "mean_leaked": float(np.mean(leaks)),
    }
