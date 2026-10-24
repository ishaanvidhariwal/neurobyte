"""Experiment drivers: Monte-Carlo comparison, sensitivity sweep, ABAB reversal."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .environment import run_episode
from .policies import AdaptiveOnline, AdaptiveStabilized, FixedOrder, StaticFrequency
from .user import CostConfig, DemandConfig, DemandProcess, SelectionCostModel, zipf_pmf

WARMUP = 60
SESSION_LEN = 40


def build_policies(n: int, warmup: int = WARMUP):
    return [
        FixedOrder(n),
        StaticFrequency(n, warmup),
        AdaptiveOnline(n),
        AdaptiveStabilized(n),
    ]


POLICY_NAMES = [p.name for p in build_policies(1)]


def run_single(seed: int, demand_cfg: DemandConfig, cost_cfg: CostConfig) -> dict:
    """One simulated user (one seed) run through every policy on the same demands."""
    demands = DemandProcess(demand_cfg, seed=seed)
    out = {}
    for pol in build_policies(demand_cfg.n_commands):
        # Same cost-model seed for every policy so the noise stream starts
        # identical; it only diverges where a policy actually causes a slip.
        out[pol.name] = run_episode(
            demands, pol, cost_cfg, seed=seed + 10_000,
            warmup=WARMUP, session_len=demand_cfg.session_len,
        )
    return out


def run_monte_carlo(n_seeds: int, demand_cfg: DemandConfig, cost_cfg: CostConfig,
                    base_seed: int = 0):
    metrics = ("mean_time", "error_rate", "corr_per_trial")
    agg = {nm: {m: [] for m in metrics} | {"session_time": []} for nm in POLICY_NAMES}
    for s in range(n_seeds):
        r = run_single(base_seed + s, demand_cfg, cost_cfg)
        for nm in POLICY_NAMES:
            for m in metrics:
                agg[nm][m].append(r[nm][m])
            agg[nm]["session_time"].append(r[nm]["session_time"])
    for nm in POLICY_NAMES:
        for k in agg[nm]:
            agg[nm][k] = np.array(agg[nm][k])
    return POLICY_NAMES, agg


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def paired_summary(agg: dict, ref: str = "static-freq", metric: str = "mean_time"):
    """Per policy: bootstrap CI of the metric, and paired difference vs ``ref``.

    Pairing is valid because seed ``s`` feeds the identical demand sequence to
    every policy, so ``agg[a][metric][s]`` and ``agg[b][metric][s]`` describe
    the same simulated user.
    """
    rows = {}
    ref_x = agg[ref][metric]
    for nm in agg:
        x = agg[nm][metric]
        m, lo, hi = bootstrap_ci(x, seed=1)
        diff = x - ref_x
        if nm == ref or diff.std(ddof=1) == 0:
            dz = 0.0
        else:
            dz = float(diff.mean() / diff.std(ddof=1))   # Cohen's dz (paired)
        dm, dlo, dhi = bootstrap_ci(diff, seed=2)
        rows[nm] = dict(mean=m, lo=lo, hi=hi, diff=dm, diff_lo=dlo, diff_hi=dhi, dz=dz)
    return rows


def sweep(param: str, values, n_seeds: int,
          base_demand: DemandConfig, base_cost: CostConfig, base_seed: int = 5000):
    """Mean selection time per policy as one parameter is swept."""
    grid = {nm: [] for nm in POLICY_NAMES}
    for v in values:
        dc, cc = replace(base_demand), replace(base_cost)
        if hasattr(dc, param):
            setattr(dc, param, v)
        elif hasattr(cc, param):
            setattr(cc, param, v)
        else:
            raise ValueError(f"unknown parameter: {param}")
        _, agg = run_monte_carlo(n_seeds, dc, cc, base_seed=base_seed)
        for nm in POLICY_NAMES:
            grid[nm].append(float(agg[nm]["mean_time"].mean()))
    return POLICY_NAMES, {nm: np.array(v) for nm, v in grid.items()}


def run_abab(seed: int, n_commands: int = 12, sessions: int = 16,
             session_len: int = SESSION_LEN, cost_cfg: CostConfig | None = None):
    """Single-subject ABAB reversal on one adaptive-stabilized policy.

    A phases: the layout is frozen (adaptation withheld) while counts keep
    updating in the background. B phases: adaptation is enabled. Popularity
    drifts by one adjacent swap every session throughout, so a frozen layout
    keeps decaying and an enabled one keeps repairing itself. If selection
    time drops in every B phase and rises again in every A phase, the layout
    adaptation -- not time or practice -- is what moved the metric.
    """
    cost_cfg = cost_cfg or CostConfig()
    rng = np.random.default_rng(seed)
    base = zipf_pmf(n_commands, 1.1)
    perm = rng.permutation(n_commands)

    cost = SelectionCostModel(cost_cfg, n_commands, seed=seed + 7)
    pol = AdaptiveStabilized(n_commands)

    phase = ["A" if (s // 4) % 2 == 0 else "B" for s in range(sessions)]
    sess_mean = np.empty(sessions)

    for s in range(sessions):
        i = int(rng.integers(0, n_commands - 1))         # session drift: one adjacent swap
        perm[[i, i + 1]] = perm[[i + 1, i]]
        p = np.zeros(n_commands)
        p[perm] = base

        adapt = phase[s] == "B"
        tvals = np.empty(session_len)
        for j in range(session_len):
            c = int(rng.choice(n_commands, p=p))
            tvals[j] = cost.select(c, pol.current_order)["time"]
            if adapt:
                pol.update(c, 0)
            else:
                pol.w *= pol.decay      # keep the estimate warm, but hold the layout
                pol.w[c] += 1.0
        sess_mean[s] = tvals.mean()

    return phase, sess_mean
