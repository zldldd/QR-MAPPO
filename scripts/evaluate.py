"""Evaluate a trained QR-MAPPO checkpoint."""
import argparse
import json
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from qr_mappo.algorithms.qr_mappo_trainer import QRMAPPOTrainer
from qr_mappo.envs.air_defense_env import AirDefenseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="S2", choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(scenario=args.scenario)
    cfg.algo.seed = args.seed
    cfg.algo.device = args.device

    env = AirDefenseEnv(cfg)
    trainer = QRMAPPOTrainer(cfg, env)
    trainer.load_checkpoint(args.checkpoint)

    metrics = trainer.evaluate(n_episodes=args.n_episodes)
    print(json.dumps(metrics, indent=2))

    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
