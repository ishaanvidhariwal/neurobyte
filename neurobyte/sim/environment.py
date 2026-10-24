"""Run one episode: a policy driven through a full demand sequence."""

from __future__ import annotations

import numpy as np

from .user import CostConfig, DemandProcess, SelectionCostModel


def run_episode(
    demands: DemandProcess,
    policy,
    cost_cfg: CostConfig,
    seed: int,
    warmup: int,
    session_len: int,
) -> dict:
    n = demands.cfg.n_commands
    cost = SelectionCostModel(cost_cfg, n, seed=seed)
    T = len(demands)

    times = np.empty(T)
    errors = np.zeros(T)
    corr = np.zeros(T)

    for t in range(T):
        c = demands[t]
        r = cost.select(c, policy.current_order)
        times[t] = r["time"]
        errors[t] = r["error"]
        corr[t] = r["corrections"]
        policy.update(c, t)

    post = slice(warmup, T)
    n_sessions = T // session_len
    keep = n_sessions * session_len
    sess_time = times[:keep].reshape(n_sessions, session_len).mean(axis=1)
    sess_err = errors[:keep].reshape(n_sessions, session_len).mean(axis=1)

    return {
        "mean_time": float(times[post].mean()),
        "error_rate": float(errors[post].mean()),
        "corr_per_trial": float(corr[post].mean()),
        "session_time": sess_time,
        "session_err": sess_err,
    }
