"""Environment sanity tests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from qr_mappo.envs.air_defense_env import AirDefenseEnv


def test_reset_and_step():
    cfg = Config(scenario="S2")
    env = AirDefenseEnv(cfg)
    obs, state, info = env.reset(seed=0)
    assert set(obs.keys()) == set(env.get_agent_ids())
    assert state.shape == (cfg.algo.state_dim,)
    assert all(o.shape == (cfg.algo.obs_dim,) for o in obs.values())

    actions = {a: env.rng.integers(0, env.action_space_size) for a in env.get_agent_ids()}
    next_obs, next_state, rewards, terms, truncs, infos = env.step(actions)
    assert set(next_obs.keys()) == set(env.get_agent_ids())
    assert all(a in rewards for a in env.get_agent_ids())
    assert all(a in terms for a in env.get_agent_ids())
    print("test_reset_and_step passed.")


def test_episode():
    cfg = Config(scenario="S1")
    env = AirDefenseEnv(cfg)
    obs, _, _ = env.reset(seed=1)
    done = False
    steps = 0
    while not done and steps < 200:
        actions = {a: env.rng.integers(0, env.action_space_size) for a in env.get_agent_ids()}
        obs, _, rewards, terms, truncs, infos = env.step(actions)
        done = any(terms.values()) or any(truncs.values())
        steps += 1
    print(f"test_episode passed after {steps} steps.")


if __name__ == "__main__":
    test_reset_and_step()
    test_episode()
