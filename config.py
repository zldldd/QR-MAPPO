"""Global configuration for QR-MAPPO air-defense experiments."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ScenarioConfig:
    name: str
    n_agents: int = 15
    n_targets: int = 6
    n_episodes: int = 1
    max_steps: int = 100
    area_size: float = 200.0
    altitude_range: Tuple[float, float] = (0.0, 15.0)
    agent_positions: Dict[str, List[float]] = field(default_factory=dict)
    target_positions: List[List[float]] = field(default_factory=list)
    target_types: List[str] = field(default_factory=lambda: ["fighter"] * 6)
    target_paths: List[List[Tuple[float, float, float, float]]] = field(
        default_factory=list
    )


SCENARIOS: Dict[str, ScenarioConfig] = {
    "S1": ScenarioConfig(
        name="S1",
        n_agents=15,
        n_targets=6,
        max_steps=100,
        area_size=200.0,
        target_types=["fighter"] * 6,
    ),
    "S2": ScenarioConfig(
        name="S2",
        n_agents=15,
        n_targets=8,
        max_steps=120,
        area_size=200.0,
        target_types=["fighter"] * 5 + ["missile"] * 3,
    ),
    "S3": ScenarioConfig(
        name="S3",
        n_agents=15,
        n_targets=10,
        max_steps=150,
        area_size=250.0,
        target_types=["fighter"] * 6 + ["missile"] * 3 + ["drone"] * 1,
    ),
    "S4": ScenarioConfig(
        name="S4",
        n_agents=15,
        n_targets=12,
        max_steps=180,
        area_size=250.0,
        target_types=["fighter"] * 7 + ["missile"] * 4 + ["drone"] * 1,
    ),
}


@dataclass
class AlgorithmConfig:
    # Training
    total_steps: int = 5_000_000
    rollout_length: int = 400
    n_epochs: int = 10
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip: bool = True
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_temp: float = 1e-4
    lr_rho: float = 3e-4
    optimizer_eps: float = 1e-5
    use_lr_scheduler: bool = True

    # MaxEnt / QRE
    init_log_temp: float = 0.0
    target_entropy_ratio: float = 0.6

    # SPAC
    spac_window: int = 50
    spac_high: float = 2.0
    spac_low: float = 0.5
    spac_delta: float = 0.05

    # PSP
    psp_lambda: float = 0.01
    psp_epsilon: float = 0.05
    psp_update_freq: int = 5

    # Network
    obs_dim: int = 450
    state_dim: int = 450
    agent_type_dim: int = 15
    agent_id_dim: int = 16
    hidden_dim: int = 256
    rnn_hidden_dim: int = 128
    comm_dim: int = 64
    n_comm_heads: int = 4

    # Evaluation
    eval_interval: int = 50_000
    save_interval: int = 100_000
    n_eval_episodes: int = 20

    # Misc
    device: str = "cpu"
    seed: int = 0


@dataclass
class EnvConfig:
    dt: float = 1.0
    max_range_radar: float = 120.0
    max_range_jammer: float = 80.0
    max_range_missile: float = 60.0
    max_range_gun: float = 8.0
    radar_fov: float = 120.0
    n_freq_channels: int = 8
    freq_min: float = 2.0
    freq_max: float = 18.0
    p_kill_base: float = 0.85
    p_kill_noise: float = 0.05
    jamming_jsr_threshold: float = 10.0
    reward_kill: float = 10.0
    reward_survive: float = 1.0
    penalty_leakage: float = -5.0
    penalty_friendly_fire: float = -3.0
    penalty_collision: float = -1.0
    penalty_resource: float = -0.05
    rew_shaping_eta: float = 1.0


@dataclass
class Config:
    scenario: str = "S2"
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def __post_init__(self):
        self.scenario_cfg = SCENARIOS[self.scenario]
