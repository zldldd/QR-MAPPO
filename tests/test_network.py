"""Network shape tests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from qr_mappo.envs.air_defense_env import AirDefenseEnv
from qr_mappo.models.networks import DualSoftCritic, QREActor, RationalityEncoder


def test_actor_critic():
    cfg = Config(scenario="S2")
    env = AirDefenseEnv(cfg)
    obs, state, _ = env.reset(seed=0)

    actor = QREActor(
        obs_dim=cfg.algo.obs_dim,
        n_agents=env.n_agents,
        n_types=5,
        action_dim=env.action_space_size,
        hidden_dim=cfg.algo.hidden_dim,
        agent_type_dim=cfg.algo.agent_type_dim,
        agent_id_dim=cfg.algo.agent_id_dim,
        comm_dim=cfg.algo.comm_dim,
        n_comm_heads=cfg.algo.n_comm_heads,
    )
    critic = DualSoftCritic(cfg.algo.state_dim, cfg.algo.hidden_dim)
    rho_encoder = RationalityEncoder(cfg.algo.state_dim, cfg.algo.hidden_dim)

    obs_t = torch.stack([torch.from_numpy(obs[a]).float() for a in env.get_agent_ids()]).unsqueeze(0)
    state_t = torch.from_numpy(state).float().unsqueeze(0)
    type_map = {"radar": 0, "jammer": 1, "missile": 2, "gun": 3, "command": 4}
    types_t = torch.tensor(
        [type_map["radar"] if i <= 4 else type_map["jammer"] if i == 5
         else type_map["missile"] if i <= 10 else type_map["gun"] if i <= 13
         else type_map["command"] for i in range(1, env.n_agents + 1)]
    ).unsqueeze(0)
    ids_t = torch.arange(env.n_agents).unsqueeze(0)

    rho, rho_embed = rho_encoder(state_t)
    assert rho.shape == (1, 1)
    assert rho_embed.shape == (1, 32)

    dist, temp = actor(obs_t, types_t, ids_t, rho_embed)
    assert dist.logits.shape == (1, env.n_agents, env.action_space_size)
    assert temp.shape == (1,)

    v1, v2 = critic(state_t)
    assert v1.shape == (1,) and v2.shape == (1,)
    print("test_actor_critic passed.")


if __name__ == "__main__":
    test_actor_critic()
