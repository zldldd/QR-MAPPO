"""Simplified heterogeneous integrated air-defense environment.

Implements a PettingZoo-style parallel multi-agent API.  All agents share the
same discrete action space: 0 = no-op, 1..n_targets = allocate attention or
weapon to the corresponding target.  Agent type-specific logic maps the action
index to behaviour (radar illumination, jamming, missile/gun engagement).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Target:
    """Attacker (air target)."""

    TYPES = {"fighter": 1.0, "missile": 0.5, "drone": 1.2}

    def __init__(
        self,
        target_id: int,
        target_type: str,
        start: np.ndarray,
        goal: np.ndarray,
        speed: float,
        max_steps: int,
    ):
        self.id = target_id
        self.target_type = target_type
        self.position = start.astype(float).copy()
        self.start = start.copy()
        self.goal = goal.copy()
        self.speed = speed
        self.max_steps = max_steps
        self.alive = True
        self.leaked = False
        self.killed = False
        self.jammed = False
        self.detected = False
        self.tracked = False
        self.engaged_by: List[int] = []
        self.inbound_missiles: List[Dict[str, Any]] = []
        self.hp = self.TYPES.get(target_type, 1.0)
        self.evasion_timer = 0

    def step(self, dt: float, adversary_rho: float = 1.0, rng: Optional[np.random.Generator] = None):
        if not self.alive or self.leaked:
            return
        rng = rng or np.random.default_rng()
        direction = self.goal - self.position
        dist = float(np.linalg.norm(direction[:2]) + 1e-8)
        if dist < 2.0:
            self.leaked = True
            self.alive = False
            return
        direction = direction / dist
        # QRE-like evasion: targets with higher threat may manoeuvre.
        if self.evasion_timer > 0:
            lateral = np.array([-direction[1], direction[0], 0.0])
            direction = direction + lateral * rng.normal(0, 0.2)
            self.evasion_timer -= 1
        elif rng.random() < 1.0 / (1.0 + math.exp(adversary_rho)):
            self.evasion_timer = max(1, int(3 / dt))
        self.position += direction * self.speed * dt
        self.position[2] = float(np.clip(self.position[2], 0.0, 15.0))


class Agent:
    """Defender asset."""

    ROLES = {
        "radar": ["A1", "A2", "A3", "A4"],
        "jammer": ["A5"],
        "missile": ["A6", "A7", "A8", "A9", "A10"],
        "gun": ["A11", "A12", "A13"],
        "command": ["A14", "A15"],
    }

    def __init__(
        self,
        agent_id: str,
        role: str,
        position: np.ndarray,
        config: Any,
    ):
        self.id = agent_id
        self.role = role
        self.position = position.astype(float).copy()
        self.config = config
        self.alive = True
        self.action = 0
        self.target_id: Optional[int] = None
        self.cooldown = 0
        self.ammo = self._starting_ammo()
        self.lock_timer = 0
        self.jam_timer = 0

    def _starting_ammo(self) -> int:
        if self.role == "missile":
            return 4
        if self.role == "gun":
            return 200
        return -1

    @property
    def type_index(self) -> int:
        return list(self.ROLES.keys()).index(self.role)


class AirDefenseEnv:
    """Parallel multi-agent environment."""

    metadata = {"render_modes": [], "name": "qr_mappo_air_defense"}

    def __init__(self, config: Any, scenario_cfg: Optional[Any] = None):
        self.cfg = config.env
        self.scenario = scenario_cfg or config.scenario_cfg
        self.rng = np.random.default_rng(config.algo.seed)
        self.rho_adv = 1.0

        self.possible_agents = [f"A{i}" for i in range(1, 16)]
        self.agents: List[Agent] = []
        self.targets: List[Target] = []
        self.step_count = 0
        self.episode_reward = 0.0

        # Action / observation spaces for the training loop.
        self.n_agents = len(self.possible_agents)
        self.n_targets = self.scenario.n_targets
        self.action_space_size = self.n_targets + 1
        self.obs_dim = config.algo.obs_dim
        self.state_dim = config.algo.state_dim

        self._init_entities()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _random_position(self, altitude: float) -> np.ndarray:
        x = self.rng.uniform(0, self.scenario.area_size)
        y = self.rng.uniform(0, self.scenario.area_size)
        return np.array([x, y, altitude], dtype=float)

    def _init_entities(self):
        self.agents = []
        for role, ids in Agent.ROLES.items():
            for aid in ids:
                pos = self._random_position(0.0 if role != "command" else 2.0)
                self.agents.append(Agent(aid, role, pos, self.cfg))
        self.agents.sort(key=lambda a: int(a.id[1:]))
        self._spawn_targets()

    def _spawn_targets(self):
        self.targets = []
        for i in range(self.n_targets):
            ttype = self.scenario.target_types[i % len(self.scenario.target_types)]
            start = self._random_position(self.rng.uniform(5, 12))
            goal = self._random_position(self.rng.uniform(0, 5))
            speed = {"fighter": 8.0, "missile": 18.0, "drone": 6.0}[ttype]
            self.targets.append(
                Target(i, ttype, start, goal, speed, self.scenario.max_steps)
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.episode_reward = 0.0
        self._init_entities()
        obs = {a.id: self._get_obs(a) for a in self.agents}
        state = self._get_state()
        info = {a.id: {} for a in self.agents}
        return obs, state, info

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray], np.ndarray, Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        self.step_count += 1
        # 1. Adversary moves targets.
        for t in self.targets:
            t.step(self.cfg.dt, self.rho_adv, self.rng)
            t.jammed = False
            t.detected = False
            t.tracked = False
            if t.inbound_missiles:
                for m in list(t.inbound_missiles):
                    m["time_left"] -= self.cfg.dt
                    if m["time_left"] <= 0:
                        self._resolve_missile(t, m)
                        t.inbound_missiles.remove(m)

        # 2. Reset agent timers.
        for a in self.agents:
            a.action = int(actions.get(a.id, 0))
            a.target_id = a.action - 1 if a.action > 0 else None
            if a.cooldown > 0:
                a.cooldown -= self.cfg.dt
            if a.lock_timer > 0:
                a.lock_timer -= self.cfg.dt
            if a.jam_timer > 0:
                a.jam_timer -= self.cfg.dt

        # 3. Radar detection.
        self._update_detection()

        # 4. Jamming.
        self._update_jamming()

        # 5. Engagements.
        self._update_engagements()

        # 6. Rewards / terminations.
        rewards, terminations, truncations, infos = self._compute_transitions()
        obs = {a.id: self._get_obs(a) for a in self.agents}
        state = self._get_state()
        return obs, state, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def _update_detection(self):
        for a in self.agents:
            if a.role != "radar" or not a.alive:
                continue
            for t in self.targets:
                if not t.alive:
                    continue
                d = float(np.linalg.norm(a.position - t.position))
                if d > self.cfg.max_range_radar:
                    continue
                # Jammed target is harder to detect.
                p = math.exp(-d / self.cfg.max_range_radar)
                if t.jammed:
                    p *= 0.3
                if self.rng.random() < p:
                    t.detected = True
                    if self.rng.random() < 0.7:
                        t.tracked = True
                    a.lock_timer = max(a.lock_timer, self.cfg.dt * 2)

    def _update_jamming(self):
        for a in self.agents:
            if a.role != "jammer" or not a.alive or a.target_id is None:
                continue
            t = self.targets[a.target_id]
            if not t.alive:
                continue
            d = float(np.linalg.norm(a.position - t.position))
            if d < self.cfg.max_range_jammer:
                t.jammed = True
                a.jam_timer = max(a.jam_timer, self.cfg.dt * 2)

    def _update_engagements(self):
        for a in self.agents:
            if a.role not in ("missile", "gun") or not a.alive or a.target_id is None:
                continue
            if a.cooldown > 0:
                continue
            t = self.targets[a.target_id]
            if not t.alive or t.id in t.engaged_by:
                continue
            d = float(np.linalg.norm(a.position - t.position))
            max_range = (
                self.cfg.max_range_missile if a.role == "missile" else self.cfg.max_range_gun
            )
            if d > max_range:
                continue
            if a.role == "missile" and a.ammo <= 0:
                continue
            if not t.tracked and self.rng.random() < 0.5:
                continue

            if a.role == "missile":
                a.ammo -= 1
                a.cooldown = 6.0
                t.engaged_by.append(a.id)
                t.inbound_missiles.append(
                    {"agent": a.id, "time_left": max(1.0, d / 15.0), "dist": d}
                )
            else:  # gun
                a.cooldown = 1.0
                p = self._kill_prob(a, t, d)
                if self.rng.random() < p:
                    t.alive = False
                    t.killed = True

    def _resolve_missile(self, t: Target, m: Dict[str, Any]):
        a_id = m["agent"]
        a = next((x for x in self.agents if x.id == a_id), None)
        d = m["dist"]
        p = self._kill_prob(a, t, d) if a else self.cfg.p_kill_base
        if self.rng.random() < p:
            t.alive = False
            t.killed = True

    def _kill_prob(self, a: Optional[Agent], t: Target, d: float) -> float:
        p = self.cfg.p_kill_base
        p *= math.exp(-d / (self.cfg.max_range_missile + 1e-8))
        if t.jammed:
            p *= 0.5
        if t.target_type == "missile":
            p *= 0.85
        if a and a.role == "gun":
            p = min(0.95, p * 1.3)
        p = float(np.clip(p, 0.05, 0.98))
        return p

    def _compute_transitions(self):
        rewards: Dict[str, float] = {a.id: 0.0 for a in self.agents}
        infos: Dict[str, Any] = {a.id: {} for a in self.agents}
        terminations: Dict[str, bool] = {a.id: False for a in self.agents}
        truncations: Dict[str, bool] = {a.id: False for a in self.agents}

        global_r = 0.0
        for t in self.targets:
            if t.killed:
                global_r += self.cfg.reward_kill
            if t.leaked:
                global_r += self.cfg.penalty_leakage

        alive_agents = sum(1 for a in self.agents if a.alive)
        global_r += alive_agents * self.cfg.reward_survive * 0.1

        # Resource penalty
        for a in self.agents:
            if a.action > 0:
                global_r += self.cfg.penalty_resource

        # Shaping: encourage detecting / tracking / jamming high-threat targets.
        for t in self.targets:
            if not t.alive:
                continue
            if t.tracked:
                global_r += 0.05
            if t.jammed:
                global_r += 0.05

        self.episode_reward += global_r
        for a in self.agents:
            rewards[a.id] = global_r / self.n_agents

        done = (
            not any(t.alive for t in self.targets)
            or all(t.leaked for t in self.targets)
            or alive_agents == 0
        )
        trunc = self.step_count >= self.scenario.max_steps
        for a in self.agents:
            terminations[a.id] = done
            truncations[a.id] = trunc
            infos[a.id] = {
                "episode_step": self.step_count,
                "alive_targets": sum(1 for t in self.targets if t.alive),
                "killed": sum(1 for t in self.targets if t.killed),
                "leaked": sum(1 for t in self.targets if t.leaked),
            }
        return rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # Observations / state
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        return self._vectorise(full_state=True)

    def _get_obs(self, agent: Agent) -> np.ndarray:
        return self._vectorise(agent=agent)

    def _vectorise(self, agent: Optional[Agent] = None, full_state: bool = False) -> np.ndarray:
        """Return a fixed-size float vector.

        If ``full_state`` is True, a global state vector is returned.
        Otherwise a per-agent observation is returned.
        """
        vec: List[float] = []
        # Global scalar features.
        vec.append(self.step_count / max(1, self.scenario.max_steps))
        vec.append(self.rho_adv / 3.0)
        vec.extend([self.scenario.area_size / 300.0, float(self.n_targets) / 15.0])

        # Agent features.
        for a in self.agents:
            role_onehot = [0.0] * 5
            role_onehot[a.type_index] = 1.0
            feats = [
                a.position[0] / self.scenario.area_size,
                a.position[1] / self.scenario.area_size,
                a.position[2] / 20.0,
                1.0 if a.alive else 0.0,
                a.cooldown / 10.0,
                (a.ammo / 4.0) if a.role == "missile" else (a.ammo / 200.0),
            ]
            vec.extend(role_onehot)
            vec.extend(feats)

        # Target features.
        for t in self.targets:
            type_onehot = [0.0] * 3
            idx = {"fighter": 0, "missile": 1, "drone": 2}.get(t.target_type, 0)
            type_onehot[idx] = 1.0
            feats = [
                t.position[0] / self.scenario.area_size,
                t.position[1] / self.scenario.area_size,
                t.position[2] / 20.0,
                1.0 if t.alive else 0.0,
                1.0 if t.detected else 0.0,
                1.0 if t.tracked else 0.0,
                1.0 if t.jammed else 0.0,
                len(t.inbound_missiles) / 5.0,
            ]
            vec.extend(type_onehot)
            vec.extend(feats)

        arr = np.array(vec, dtype=np.float32)
        if full_state:
            target_size = self.state_dim
        else:
            target_size = self.obs_dim
        if arr.size < target_size:
            arr = np.concatenate([arr, np.zeros(target_size - arr.size, dtype=np.float32)])
        return arr[:target_size]

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def set_adversary_rationality(self, rho: float):
        self.rho_adv = float(rho)

    def get_agent_ids(self) -> List[str]:
        return [a.id for a in self.agents]

    def render(self):
        pass

    def close(self):
        pass
