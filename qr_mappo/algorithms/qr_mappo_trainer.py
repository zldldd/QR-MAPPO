"""QR-MAPPO trainer: CTDE PPO + GAE + MaxEnt + SPAC + PSP."""
from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.networks import DualSoftCritic, ParameterSpacePerturbation, QREActor, RationalityEncoder
from ..utils.spac import SPACScheduler
from .rollout_buffer import RolloutBuffer


class QRMAPPOTrainer:
    def __init__(self, config: Any, env: Any):
        self.cfg = config.algo
        self.env_cfg = config.env
        self.env = env
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.")

        self.n_agents = env.n_agents
        self.action_dim = env.action_space_size
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim

        # Networks
        self.actor = QREActor(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_types=5,
            action_dim=self.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            agent_type_dim=self.cfg.agent_type_dim,
            agent_id_dim=self.cfg.agent_id_dim,
            comm_dim=self.cfg.comm_dim,
            n_comm_heads=self.cfg.n_comm_heads,
            use_comm=True,
            use_rho=True,
        ).to(self.device)

        self.critic = DualSoftCritic(self.state_dim, self.cfg.hidden_dim).to(self.device)
        self.rho_encoder = RationalityEncoder(self.state_dim, self.cfg.hidden_dim).to(self.device)

        # Optimizers
        self.actor_optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.rho_encoder.parameters()),
            lr=self.cfg.lr_actor,
            eps=self.cfg.optimizer_eps,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic, eps=self.cfg.optimizer_eps
        )

        if self.cfg.use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
                self.actor_optim, step_size=200, gamma=0.95
            )
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
                self.critic_optim, step_size=200, gamma=0.95
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        self.buffer = RolloutBuffer(
            rollout_length=self.cfg.rollout_length,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            n_types=5,
            action_dim=self.action_dim,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=self.device,
        )

        self.spac = SPACScheduler(
            window=self.cfg.spac_window,
            rho_high=self.cfg.spac_high,
            rho_low=self.cfg.spac_low,
            delta=self.cfg.spac_delta,
        )
        self.psp = ParameterSpacePerturbation(epsilon=self.cfg.psp_epsilon)

        self.total_steps = 0
        self.update_count = 0
        self.episode_returns: List[float] = []
        self.writer = SummaryWriter(log_dir=f"runs/qr_mappo_{config.scenario}_seed{self.cfg.seed}")

    # ------------------------------------------------------------------ #
    # Interaction helpers
    # ------------------------------------------------------------------ #
    def _prepare_obs(self, obs_dict: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_list = [obs_dict[a] for a in self.env.get_agent_ids()]
        obs = np.stack(obs_list, axis=0)
        types = np.array([self._agent_type_index(a) for a in self.env.get_agent_ids()], dtype=np.int64)
        ids = np.array([int(a[1:]) - 1 for a in self.env.get_agent_ids()], dtype=np.int64)
        return (
            torch.from_numpy(obs).unsqueeze(0).float().to(self.device),
            torch.from_numpy(types).unsqueeze(0).to(self.device),
            torch.from_numpy(ids).unsqueeze(0).to(self.device),
        )

    def _agent_type_index(self, agent_id: str) -> int:
        mapping = {"radar": 0, "jammer": 1, "missile": 2, "gun": 3, "command": 4}
        # Determine by id range based on the ROLES mapping.
        idx = int(agent_id[1:])
        if idx <= 4:
            return mapping["radar"]
        if idx == 5:
            return mapping["jammer"]
        if idx <= 10:
            return mapping["missile"]
        if idx <= 13:
            return mapping["gun"]
        return mapping["command"]

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    def collect_rollouts(self):
        self.buffer.reset()
        obs_dict, state_vec, _ = self.env.reset(seed=self.cfg.seed + self.total_steps)
        state = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)

        episode_return = 0.0
        for _ in range(self.cfg.rollout_length):
            obs, agent_types, agent_ids = self._prepare_obs(obs_dict)
            with torch.no_grad():
                rho_hat, rho_embed = self.rho_encoder(state)
                dist, temp = self.actor(obs, agent_types, agent_ids, rho_embed)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                v1, v2 = self.critic(state)
                value = v1.squeeze(-1).cpu().numpy()

            action_np = action.squeeze(0).cpu().numpy()
            log_prob_np = log_prob.squeeze(0).cpu().numpy()
            value_np = np.full(self.n_agents, float(value), dtype=np.float32)

            next_obs_dict, next_state_vec, rewards, terms, truncs, infos = self.env.step(
                {a: int(action_np[i]) for i, a in enumerate(self.env.get_agent_ids())}
            )
            reward_np = np.array([rewards[a] for a in self.env.get_agent_ids()], dtype=np.float32)
            done_np = np.array(
                [float(terms[a] or truncs[a]) for a in self.env.get_agent_ids()],
                dtype=np.float32,
            )
            types_np = agent_types.squeeze(0).cpu().numpy()
            ids_np = agent_ids.squeeze(0).cpu().numpy()

            self.buffer.add(
                obs.squeeze(0).cpu().numpy(),
                state.squeeze(0).cpu().numpy(),
                action_np,
                log_prob_np,
                value_np,
                reward_np,
                done_np,
                types_np,
                ids_np,
            )

            episode_return += float(reward_np.mean())
            self.total_steps += 1

            if any(terms.values()) or any(truncs.values()):
                self.episode_returns.append(episode_return)
                episode_return = 0.0
                obs_dict, state_vec, _ = self.env.reset(seed=self.cfg.seed + self.total_steps)
                state = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            else:
                obs_dict = next_obs_dict
                state = torch.from_numpy(next_state_vec).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            obs, agent_types, agent_ids = self._prepare_obs(obs_dict)
            rho_hat, rho_embed = self.rho_encoder(state)
            _, _ = self.actor(obs, agent_types, agent_ids, rho_embed)
            v1, v2 = self.critic(state)
            next_value = v1.squeeze(-1).cpu().numpy()
        self.buffer.compute_returns_and_advantages(next_value)

        if self.episode_returns:
            recent = np.mean(self.episode_returns[-self.cfg.spac_window :])
            self.env.set_adversary_rationality(self.spac.update(recent))

    # ------------------------------------------------------------------ #
    # Learning update
    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        self.update_count += 1
        data = {
            "obs": torch.from_numpy(self.buffer.obs).to(self.device),
            "states": torch.from_numpy(self.buffer.states).to(self.device),
            "actions": torch.from_numpy(self.buffer.actions).to(self.device),
            "log_probs": torch.from_numpy(self.buffer.log_probs).to(self.device),
            "returns": torch.from_numpy(self.buffer.returns).to(self.device),
            "advantages": torch.from_numpy(self.buffer.advantages).to(self.device),
            "agent_types": torch.from_numpy(self.buffer.agent_types).to(self.device),
            "agent_ids": torch.from_numpy(self.buffer.agent_ids).to(self.device),
        }

        T, N = self.cfg.rollout_length, self.n_agents
        obs_b = data["obs"].reshape(T * N, -1)
        states_b = data["states"].reshape(T, -1)
        actions_b = data["actions"].reshape(T * N)
        old_log_probs_b = data["log_probs"].reshape(T * N)
        returns_b = data["returns"].reshape(T * N)
        advantages_b = data["advantages"].reshape(T * N)
        agent_types_b = data["agent_types"].reshape(T * N)
        agent_ids_b = data["agent_ids"].reshape(T * N)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_temp_loss = 0.0
        total_psp_loss = 0.0

        # Build a perturbed snapshot of the actor once per update for PSP.
        actor_pert = None
        if self.update_count % self.cfg.psp_update_freq == 0:
            actor_pert = copy.deepcopy(self.actor)
            self.psp.perturb(actor_pert)
            actor_pert.eval()

        for batch in self.buffer.get_batches(self.cfg.batch_size, self.cfg.n_epochs):
            b_obs = batch["obs"]
            b_states = batch["states"]
            b_actions = batch["actions"]
            b_old_logp = batch["log_probs"]
            b_returns = batch["returns"]
            b_advantages = batch["advantages"]
            b_types = batch["agent_types"]
            b_ids = batch["agent_ids"]

            # Actor forward
            rho_hat, rho_embed = self.rho_encoder(b_states)
            B = b_obs.shape[0]
            rho_embed = rho_embed.reshape(B // N, N, -1)
            obs_2d = b_obs.reshape(B // N, N, -1)
            types_2d = b_types.reshape(B // N, N)
            ids_2d = b_ids.reshape(B // N, N)
            actions_2d = b_actions.reshape(B // N, N)
            old_logp_2d = b_old_logp.reshape(B // N, N)
            advantages_2d = b_advantages.reshape(B // N, N)

            dist, temp = self.actor(obs_2d, types_2d, ids_2d, rho_embed)
            log_prob = dist.log_prob(actions_2d)
            ratio = torch.exp(log_prob - old_logp_2d)
            surr1 = ratio * advantages_2d
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages_2d
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            target_entropy = -self.cfg.target_entropy_ratio * math.log(self.action_dim)
            temp_loss = -self.actor.log_temp * (entropy.detach() + target_entropy)

            # Critic update on the same batch
            v1, v2 = self.critic(b_states)
            v1_clipped = v1 if not self.cfg.value_clip else torch.clamp(
                v1, b_returns - self.cfg.clip_eps, b_returns + self.cfg.clip_eps
            )
            critic_loss = 0.5 * (
                F.mse_loss(v1, b_returns) + F.mse_loss(v2, b_returns)
            )

            # PSP: regularise current policy against a perturbed snapshot.
            psp_loss = torch.tensor(0.0, device=self.device)
            if actor_pert is not None:
                with torch.no_grad():
                    dist_pert = actor_pert(obs_2d, types_2d, ids_2d, rho_embed.detach())[0]
                psp_loss = kl_divergence(dist, dist_pert).mean()

            # Rationality encoder supervised signal against current SPAC rho.
            rho_target = torch.tensor(self.env.rho_adv, device=self.device, dtype=torch.float32)
            rho_loss = F.mse_loss(rho_hat.squeeze(-1), rho_target.expand_as(rho_hat.squeeze(-1)))

            actor_loss = (
                policy_loss
                - self.cfg.entropy_coef * entropy
                + temp_loss
                + self.cfg.psp_lambda * psp_loss
                + 0.1 * rho_loss
            )

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.rho_encoder.parameters(), self.cfg.max_grad_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_optim.step()

            total_actor_loss += policy_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_temp_loss += temp_loss.item()
            total_psp_loss += psp_loss.item()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        n_batches = (T * N / self.cfg.batch_size) * self.cfg.n_epochs
        metrics = {
            "actor_loss": total_actor_loss / max(1, n_batches),
            "critic_loss": total_critic_loss / max(1, n_batches),
            "entropy": total_entropy / max(1, n_batches),
            "temperature": temp.item(),
            "rho_adv": self.env.rho_adv,
            "rho_hat": float(rho_hat.mean().item()),
        }
        for k, v in metrics.items():
            self.writer.add_scalar(f"train/{k}", v, self.total_steps)
        return metrics

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self, total_steps: Optional[int] = None):
        total_steps = total_steps or self.cfg.total_steps
        pbar = tqdm(total=total_steps, initial=self.total_steps, desc="QR-MAPPO")
        while self.total_steps < total_steps:
            self.collect_rollouts()
            metrics = self.update()
            pbar.update(self.cfg.rollout_length)
            pbar.set_postfix(
                {
                    "return": np.mean(self.episode_returns[-10:]) if self.episode_returns else 0.0,
                    "rho": f"{metrics['rho_adv']:.2f}",
                    "temp": f"{metrics['temperature']:.3f}",
                }
            )
            if self.total_steps % self.cfg.save_interval < self.cfg.rollout_length:
                self.save_checkpoint("latest.pt")
        pbar.close()
        self.save_checkpoint("final.pt")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        returns = []
        kills = []
        leaks = []
        for ep in range(n_episodes):
            obs_dict, state_vec, _ = self.env.reset(seed=self.cfg.seed + 10000 + ep)
            state = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            ep_return = 0.0
            done = False
            while not done:
                obs, agent_types, agent_ids = self._prepare_obs(obs_dict)
                rho_hat, rho_embed = self.rho_encoder(state)
                dist, _ = self.actor(obs, agent_types, agent_ids, rho_embed)
                action = dist.sample()
                action_np = action.squeeze(0).cpu().numpy()
                obs_dict, state_vec, rewards, terms, truncs, infos = self.env.step(
                    {a: int(action_np[i]) for i, a in enumerate(self.env.get_agent_ids())}
                )
                ep_return += float(np.mean([rewards[a] for a in self.env.get_agent_ids()]))
                done = any(terms.values()) or any(truncs.values())
                state = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            returns.append(ep_return)
            kills.append(infos[self.env.get_agent_ids()[0]]["killed"])
            leaks.append(infos[self.env.get_agent_ids()[0]]["leaked"])
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_killed": float(np.mean(kills)),
            "mean_leaked": float(np.mean(leaks)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, filename: str, checkpoint_dir: str = "checkpoints/qr_mappo"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, filename)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "rho_encoder": self.rho_encoder.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "total_steps": self.total_steps,
                "update_count": self.update_count,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.rho_encoder.load_state_dict(ckpt["rho_encoder"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.update_count = ckpt.get("update_count", 0)
