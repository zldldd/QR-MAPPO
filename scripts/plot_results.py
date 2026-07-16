"""Generate TPAMI-style figures from `results/experiment_results.json`."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

RESULTS_PATH = "results/experiment_results.json"
FIG_DIR = "results/figures"

COLORS = sns.color_palette("tab10", 8)


def load():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_success_rate(data):
    algs = list(data["algorithms"].keys())
    scenarios = ["S1", "S2", "S3", "S4"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scenarios))
    width = 0.1
    for i, alg in enumerate(algs):
        vals = [data["algorithms"][alg][s]["success_rate"] for s in scenarios]
        ax.bar(x + (i - len(algs) / 2) * width, vals, width, label=alg, color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Interception success rate across scenarios")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "fig1_success_rate.png")


def plot_learning_curves(data):
    curves = data["convergence"]
    steps = np.array(curves["steps"]) / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, alg in enumerate(curves):
        if alg == "steps":
            continue
        ax.plot(steps, curves[alg], label=alg, linewidth=2, color=COLORS[i])
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Episode return")
    ax.set_title("Learning curves on scenario S2")
    ax.legend()
    save(fig, "fig2_learning_curves.png")


def plot_ablations(data):
    abl = data["ablations"]
    labels = list(abl.keys())
    success = [abl[k]["success_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, success, color=COLORS[: len(labels)])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation study on scenario S2")
    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "fig3_ablations.png")


def plot_cross_rationality(data):
    crc = data["cross_rationality"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, alg in enumerate(crc):
        if alg == "rho_test":
            continue
        ax.plot(crc["rho_test"], crc[alg], marker="o", label=alg, linewidth=2, color=COLORS[i])
    ax.set_xlabel(r"Adversary rationality $\rho$")
    ax.set_ylabel("Success rate")
    ax.set_title("Cross-rationality generalisation")
    ax.legend()
    save(fig, "fig4_cross_rationality.png")


def plot_wta(data):
    wta = data["wta"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    x = np.arange(len(wta["scenarios"]))
    ax.bar(x - 0.2, wta["wta_quality"], 0.4, label="WTA quality", color=COLORS[0])
    ax2.bar(x + 0.2, wta["runtime_seconds"], 0.4, label="Runtime (s)", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(wta["scenarios"])
    ax.set_ylabel("WTA quality", color=COLORS[0])
    ax2.set_ylabel("Runtime (s)", color=COLORS[1])
    ax.set_ylim(0, 1.0)
    ax.set_title("WTA solution quality vs runtime")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save(fig, "fig5_wta_runtime.png")


def plot_scalability(data):
    sc = data["scalability"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sc["n_agents"], sc["qr_mappo_return"], marker="o", label="QR-MAPPO", linewidth=2, color=COLORS[0])
    ax.plot(sc["n_agents"], sc["mappo_return"], marker="s", label="MAPPO", linewidth=2, color=COLORS[1])
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Episode return")
    ax.set_title("Scalability on scenario S2")
    ax.legend()
    save(fig, "fig6_scalability.png")


def main():
    data = load()
    plot_success_rate(data)
    plot_learning_curves(data)
    plot_ablations(data)
    plot_cross_rationality(data)
    plot_wta(data)
    plot_scalability(data)


if __name__ == "__main__":
    main()
