"""Train QR-MAPPO on a selected air-defense scenario."""
import argparse
import json
import os
import random

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from qr_mappo.algorithms.qr_mappo_trainer import QRMAPPOTrainer
from qr_mappo.envs.air_defense_env import AirDefenseEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="S2", choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(scenario=args.scenario)
    cfg.algo.seed = args.seed
    cfg.algo.device = args.device
    if args.steps is not None:
        cfg.algo.total_steps = args.steps

    set_seed(args.seed)
    env = AirDefenseEnv(cfg)
    trainer = QRMAPPOTrainer(cfg, env)

    exp_name = f"qr_mappo_{args.scenario}_seed{args.seed}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/{exp_name}"

    print(f"Training QR-MAPPO on {args.scenario} for {cfg.algo.total_steps} steps (seed={args.seed}).")
    trainer.train(total_steps=cfg.algo.total_steps)

    trainer.save_checkpoint("final.pt", checkpoint_dir=args.checkpoint_dir)
    eval_metrics = trainer.evaluate(n_episodes=cfg.algo.n_eval_episodes)
    print("Evaluation:", eval_metrics)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{exp_name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
