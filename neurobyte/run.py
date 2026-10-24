#!/usr/bin/env python3
"""Run the NEUROBYTE menu-adaptation simulation and write results/figures.

    python run.py                 # full run (200 seeds) + sweeps + ABAB
    python run.py --quick         # 40 seeds, coarse sweeps (fast smoke test)

Two scenarios are evaluated:

    step  -- one abrupt change in which commands are popular (the "killer
             experiment" the NEUROBYTE proposal sketches).
    walk  -- popularity drifts a little every session and never settles
             (a genuinely non-stationary world).

Outputs land in results/: one learning-curve + one summary-bar figure per
scenario, a shared sensitivity figure, the ABAB reversal figure, plus
summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.experiment import (
    WARMUP,
    paired_summary,
    run_abab,
    run_monte_carlo,
    sweep,
)
from sim.user import CostConfig, DemandConfig

RESULTS = Path(__file__).parent / "results"
LABELS = {
    "fixed": "Fixed order (control)",
    "static-freq": "Static frequency (baseline)",
    "adaptive-online": "Adaptive online",
    "adaptive-stable": "Adaptive stabilized",
}
COLORS = {
    "fixed": "#9aa0a6",
    "static-freq": "#1a73e8",
    "adaptive-online": "#e8710a",
    "adaptive-stable": "#188038",
}
SCENARIOS = {
    # small device menu: a handful of commands, one abrupt change of habits
    "step": DemandConfig(drift_mode="step", n_commands=12, n_trials=600),
    # large evolving command set (think command palette): popularity never settles
    "walk": DemandConfig(drift_mode="walk", n_commands=40, n_trials=1000,
                         zipf_s=1.0, walk_swaps_per_session=4.0),
}


def sparkline(values: np.ndarray) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return blocks[0] * len(values)
    idx = ((values - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in idx)


def fig_learning_curves(names, agg, cfg, path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for nm in names:
        st = agg[nm]["session_time"]            # (seeds, sessions)
        mean = st.mean(axis=0)
        lo = np.percentile(st, 2.5, axis=0)
        hi = np.percentile(st, 97.5, axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=LABELS[nm], color=COLORS[nm], lw=2)
        ax.fill_between(x, lo, hi, color=COLORS[nm], alpha=0.12)
    warm_end = WARMUP / cfg.session_len + 0.5
    ax.axvspan(0.5, warm_end, color="#000", alpha=0.05)
    ax.text(0.6, ax.get_ylim()[0], " warm-up", fontsize=9, va="bottom", color="#555")
    if cfg.drift_mode == "step":
        ds = cfg.drift_trial / cfg.session_len + 0.5
        ax.axvline(ds, color="#c0392b", ls="--", lw=1.2)
        ax.text(ds + 0.1, ax.get_ylim()[1], " concept drift",
                color="#c0392b", va="top", fontsize=9)
    ax.set_xlabel("session")
    ax.set_ylabel("mean selection time (s)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_summary_bars(names, rows, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [
        ("mean selection time (s)", rows["mean_time"]),
        ("error rate (slips / trial)", rows["error_rate"]),
        ("corrections / trial", rows["corr_per_trial"]),
    ]
    for ax, (sub, r) in zip(axes, specs):
        xs = np.arange(len(names))
        means = np.array([r[nm]["mean"] for nm in names])
        lo = np.array([r[nm]["lo"] for nm in names])
        hi = np.array([r[nm]["hi"] for nm in names])
        ax.bar(xs, means, color=[COLORS[nm] for nm in names],
               yerr=[means - lo, hi - means], capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[nm].split(" (")[0].replace(" ", "\n") for nm in names],
                           fontsize=8)
        ax.set_title(sub, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sensitivity(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (xlabel, xs, names, grid) in zip(axes, panels):
        for nm in names:
            ax.plot(xs, grid[nm], marker="o", ms=4, label=LABELS[nm], color=COLORS[nm])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean selection time (s)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensitivity: which regime favours which policy")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_abab(phase, curves, path):
    """curves: (seeds, sessions) array of per-session mean selection time."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(1, curves.shape[1] + 1)
    mean = curves.mean(axis=0)
    lo = np.percentile(curves, 2.5, axis=0)
    hi = np.percentile(curves, 97.5, axis=0)
    ax.plot(x, mean, "-o", color=COLORS["adaptive-stable"], lw=2, zorder=3)
    ax.fill_between(x, lo, hi, color=COLORS["adaptive-stable"], alpha=0.15)
    for i, ph in enumerate(phase):
        if ph == "A":
            ax.axvspan(i + 0.5, i + 1.5, color="#1a73e8", alpha=0.10)
    ax.set_xlabel("session")
    ax.set_ylabel("mean selection time (s)")
    ax.set_title("ABAB reversal — shaded = A (layout frozen), clear = B (adaptation on)\n"
                 "mean & 95% band over seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_scenario(key, cfg, n_seeds):
    print(f"\n[{key}] Monte-Carlo: {n_seeds} users x 4 policies x {cfg.n_trials} trials "
          f"(n_commands={cfg.n_commands}, drift={cfg.drift_mode}) ...")
    names, agg = run_monte_carlo(n_seeds, cfg, CostConfig())
    rows = {m: paired_summary(agg, ref="static-freq", metric=m)
            for m in ("mean_time", "error_rate", "corr_per_trial")}

    hdr = f"{'policy':<28}{'time s':>9}{'Δ vs base (s, 95% CI)':>26}{'dz':>7}{'err/trl':>9}"
    print(hdr)
    print("-" * len(hdr))
    for nm in names:
        rt, re = rows["mean_time"][nm], rows["error_rate"][nm]
        d = f"{rt['diff']:+.3f} [{rt['diff_lo']:+.3f},{rt['diff_hi']:+.3f}]"
        print(f"{LABELS[nm]:<28}{rt['mean']:>9.3f}{d:>26}{rt['dz']:>7.2f}{re['mean']:>9.4f}")
    for nm in names:
        print(f"  {LABELS[nm]:<28} curve {sparkline(agg[nm]['session_time'].mean(axis=0))}")

    fig_learning_curves(names, agg, cfg, RESULTS / f"fig_{key}_learning_curves.png",
                        f"[{key}] per-session selection time (mean, 95% band)")
    fig_summary_bars(names, rows, RESULTS / f"fig_{key}_summary_bars.png",
                     f"[{key}] post-warm-up metrics, mean & 95% bootstrap CI")
    return names, agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast, low-resolution smoke run")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    n_seeds = 40 if args.quick else 200
    sweep_seeds = 20 if args.quick else 60

    scen_out = {}
    for key, cfg in SCENARIOS.items():
        names, agg, rows = run_scenario(key, cfg, n_seeds)
        scen_out[key] = (names, agg, rows)

    # ---- sensitivity sweeps -------------------------------------------
    step_cfg, walk_cfg = SCENARIOS["step"], SCENARIOS["walk"]
    coarse = args.quick

    print("\nSweep: drift_strength (step scenario) ...")
    ds_vals = [0.0, 0.5, 1.0] if coarse else [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    _, ds_grid = sweep("drift_strength", ds_vals, sweep_seeds, step_cfg, CostConfig())

    print("Sweep: p_slip (step scenario) ...")
    ps_vals = [0.0, 0.35, 0.7] if coarse else [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    _, ps_grid = sweep("p_slip", ps_vals, sweep_seeds, step_cfg, CostConfig())

    print("Sweep: walk_swaps_per_session (walk scenario) ...")
    ws_vals = [0.0, 4.0, 12.0] if coarse else [0.0, 1.0, 3.0, 6.0, 10.0, 16.0, 24.0]
    _, ws_grid = sweep("walk_swaps_per_session", ws_vals, sweep_seeds, walk_cfg, CostConfig())

    fig_sensitivity(
        [("step drift strength (0 = stationary)", ds_vals, names, ds_grid),
         ("slip probability p_slip", ps_vals, names, ps_grid),
         ("walk: popularity swaps / session", ws_vals, names, ws_grid)],
        RESULTS / "fig_sensitivity.png",
    )

    # ---- ABAB reversal ----------------------------------------------------
    n_abab = 40 if args.quick else 200
    print(f"\nABAB reversal ({n_abab} seeds) ...")
    curves = []
    for s in range(n_abab):
        phase, sm = run_abab(seed=100 + s)
        curves.append(sm)
    curves = np.array(curves)
    abab_mean = curves.mean(axis=0)
    a_mean = float(np.mean([m for m, p in zip(abab_mean, phase) if p == "A"]))
    b_mean = float(np.mean([m for m, p in zip(abab_mean, phase) if p == "B"]))
    print(f"  A phases (frozen)  mean = {a_mean:.3f} s")
    print(f"  B phases (adapt)   mean = {b_mean:.3f} s   Δ = {b_mean - a_mean:+.3f} s")
    fig_abab(phase, curves, RESULTS / "fig_abab_reversal.png")

    # ---- summaries ---------------------------------------------------
    summary = {"scenarios": {}, "sensitivity": {}, "abab": {}}
    for key, (names, agg, rows) in scen_out.items():
        cfg = SCENARIOS[key]
        summary["scenarios"][key] = {
            "config": vars(cfg),
            "policies": {
                nm: {
                    "mean_time": rows["mean_time"][nm]["mean"],
                    "mean_time_ci": [rows["mean_time"][nm]["lo"], rows["mean_time"][nm]["hi"]],
                    "delta_vs_static_freq": rows["mean_time"][nm]["diff"],
                    "delta_ci": [rows["mean_time"][nm]["diff_lo"], rows["mean_time"][nm]["diff_hi"]],
                    "cohens_dz": rows["mean_time"][nm]["dz"],
                    "error_rate": rows["error_rate"][nm]["mean"],
                    "corr_per_trial": rows["corr_per_trial"][nm]["mean"],
                }
                for nm in names
            },
        }
    summary["sensitivity"] = {
        "drift_strength": {"x": ds_vals, **{nm: ds_grid[nm].tolist() for nm in names}},
        "p_slip": {"x": ps_vals, **{nm: ps_grid[nm].tolist() for nm in names}},
        "walk_swaps_per_session": {"x": ws_vals, **{nm: ws_grid[nm].tolist() for nm in names}},
    }
    summary["abab"] = {"phase": phase, "session_mean": abab_mean.tolist(),
                       "A_mean": a_mean, "B_mean": b_mean}
    summary["meta"] = {"n_seeds": n_seeds, "sweep_seeds": sweep_seeds, "warmup": WARMUP}

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, RESULTS / "summary.md")
    print(f"\nWrote figures + summary.json + summary.md to {RESULTS}/")


def write_markdown(summary, path):
    m = summary["meta"]
    lines = [
        "# NEUROBYTE simulation results",
        "",
        f"{m['n_seeds']} simulated users per scenario; {m['sweep_seeds']} per sweep point. "
        f"All numbers are properties of the cost model in `sim/user.py`, not "
        f"measurements of human behaviour.",
    ]
    for key, sc in summary["scenarios"].items():
        cfg = sc["config"]
        lines += [
            "",
            f"## Scenario `{key}` "
            f"(n_commands={cfg['n_commands']}, drift_mode={cfg['drift_mode']})",
            "",
            "| policy | time (s) | 95% CI | Δ vs baseline (s) | Cohen's dz | err/trial |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for nm, d in sc["policies"].items():
            lines.append(
                f"| {LABELS[nm]} | {d['mean_time']:.3f} | "
                f"[{d['mean_time_ci'][0]:.3f}, {d['mean_time_ci'][1]:.3f}] | "
                f"{d['delta_vs_static_freq']:+.3f} "
                f"[{d['delta_ci'][0]:+.3f}, {d['delta_ci'][1]:+.3f}] | "
                f"{d['cohens_dz']:+.2f} | {d['error_rate']:.4f} |"
            )
    ab = summary["abab"]
    lines += [
        "",
        "## ABAB reversal (adaptive-stabilized only)",
        "",
        f"- A phases (layout frozen): **{ab['A_mean']:.3f} s**",
        f"- B phases (adaptation on): **{ab['B_mean']:.3f} s** "
        f"(Δ = {ab['B_mean'] - ab['A_mean']:+.3f} s)",
        "",
        "## Figures",
        "",
        "- `fig_step_learning_curves.png`, `fig_step_summary_bars.png`",
        "- `fig_walk_learning_curves.png`, `fig_walk_summary_bars.png`",
        "- `fig_sensitivity.png` — the crossover: which regime favours which policy",
        "- `fig_abab_reversal.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
