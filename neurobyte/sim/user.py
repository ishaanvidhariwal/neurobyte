"""Simulated single user for the NEUROBYTE menu-adaptation study.

The user is modelled in two independent parts:

1. ``DemandProcess`` -- a deterministic, seeded stream of *which* command the
   user wants on each trial. Its popularity distribution is Zipfian (command
   usage in real software is heavy-tailed) and it undergoes a concept-drift
   event partway through the study (the user's task changes, so a different
   set of commands becomes popular). It does not depend on the menu policy,
   so every policy is scored on the exact same sequence of demands.

2. ``SelectionCostModel`` -- given the command the user wants and the current
   menu ordering, how long selection takes and whether the user slips. It
   encodes three qualitative findings from the adaptive-menu literature:

     * linear visual scan from the top of the menu (cost grows with the
       target's list position);
     * spatial memory -- repeatedly finding a command at a *stable* position
       collapses selection time toward a pointing-time floor, independent of
       list position;
     * relocation errors -- if a command moved since the user last selected
       it, they may "slip" to the old location and pay a correction penalty.
       Spatial memory is what makes this happen, so a menu that never moves
       never causes slips and a menu that constantly reshuffles causes many.

None of these numbers are calibrated to real humans. The point of the model
is to let the simulation show *in which regime* online adaptation helps, not
to assert that it does. See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def zipf_pmf(n: int, s: float) -> np.ndarray:
    """Normalised Zipf(``s``) probability mass over ``n`` ranked items."""
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / ranks**s
    return w / w.sum()


def _repair_permutation(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Turn an array of values in [0, n) into a valid permutation.

    First occurrence of each value is kept; later duplicates are replaced with
    the values that are missing, in random order.
    """
    out = arr.copy()
    present = set(int(v) for v in out.tolist())
    missing = [x for x in range(n) if x not in present]
    rng.shuffle(missing)
    seen: set[int] = set()
    mi = 0
    for i in range(n):
        v = int(out[i])
        if v in seen:
            out[i] = missing[mi]
            mi += 1
        else:
            seen.add(v)
    return out


@dataclass
class DemandConfig:
    n_commands: int = 12
    n_trials: int = 600
    session_len: int = 40
    zipf_s: float = 1.1
    drift_mode: str = "step"      # "step" | "walk" | "none"
    # --- step mode: one abrupt change of which commands are popular ---
    drift_trial: int = 300        # first trial index drawn from the post-drift distribution
    drift_strength: float = 1.0   # 0.0 = no change at drift, 1.0 = popularity fully re-permuted
    # --- walk mode: popularity ranking random-walks every session ---
    walk_swaps_per_session: float = 2.0


class DemandProcess:
    """Deterministic (seeded) stream of desired commands, one per trial.

    ``drift_mode``:
      * ``"none"`` -- stationary popularity for the whole run.
      * ``"step"`` -- one abrupt re-permutation of popularity at ``drift_trial``.
      * ``"walk"`` -- popularity ranking drifts by a few adjacent swaps every
        session (a non-stationary world that never settles).
    """

    def __init__(self, cfg: DemandConfig, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        n = cfg.n_commands
        base = zipf_pmf(n, cfg.zipf_s)

        perm_a = rng.permutation(n)                      # initial popularity ranking
        fresh = rng.permutation(n)                       # a fully independent ranking
        take_fresh = rng.random(n) < cfg.drift_strength  # per rank-slot: adopt the fresh ranking?
        perm_b = np.where(take_fresh, fresh, perm_a)
        perm_b = _repair_permutation(perm_b, n, rng)     # post-step-drift ranking

        self.p_pre = np.zeros(n)
        self.p_pre[perm_a] = base
        self.p_post = np.zeros(n)
        self.p_post[perm_b] = base

        self.demands = np.empty(cfg.n_trials, dtype=int)
        perm = perm_a.copy()
        frac_carry = 0.0
        for t in range(cfg.n_trials):
            if cfg.drift_mode == "walk" and t > 0 and t % cfg.session_len == 0:
                frac_carry += cfg.walk_swaps_per_session
                n_swaps = int(frac_carry)
                frac_carry -= n_swaps
                for _ in range(n_swaps):
                    i = int(rng.integers(0, n - 1))
                    perm[[i, i + 1]] = perm[[i + 1, i]]
                p = np.zeros(n)
                p[perm] = base
            elif cfg.drift_mode == "step":
                p = self.p_pre if t < cfg.drift_trial else self.p_post
            elif cfg.drift_mode == "none":
                p = self.p_pre
            elif cfg.drift_mode == "walk":
                p = np.zeros(n)
                p[perm] = base
            self.demands[t] = rng.choice(n, p=p)

    def __len__(self) -> int:
        return self.cfg.n_trials

    def __getitem__(self, t: int) -> int:
        return int(self.demands[t])


@dataclass
class CostConfig:
    a_scan: float = 0.30       # fixed selection overhead (s)
    b_scan: float = 0.12       # extra scan time per menu position (s)
    t_point: float = 0.45      # selection time once the position is known by heart (s)
    c_move: float = 0.03       # pointer travel time per menu position (s)
    fam_gain: float = 0.10     # spatial-memory gained per stable repeat
    fam_cap: float = 0.85      # ceiling on spatial memory (search never fully vanishes)
    fam_decay: float = 0.55    # spatial-memory retained when the item moves
    p_slip: float = 0.35       # slip probability when a fully-familiar item has moved
    t_correction: float = 1.2  # penalty for slipping to an item's old location (s)
    noise_sd: float = 0.08     # per-selection Gaussian noise (s, half-normal so >= 0)


class SelectionCostModel:
    """Stateful cost of one selection. Reset (new instance) per policy per run."""

    def __init__(self, cfg: CostConfig, n_commands: int, seed: int):
        self.cfg = cfg
        self.n = n_commands
        self.rng = np.random.default_rng(seed)
        self.familiarity = np.zeros(n_commands)     # spatial memory per command, in [0, 1]
        self.last_pos = -np.ones(n_commands, dtype=int)

    def select(self, command: int, menu_order: list[int]) -> dict:
        cfg = self.cfg
        k = menu_order.index(command)               # 0-based position of the target
        f = float(self.familiarity[command])

        scan = cfg.a_scan + cfg.b_scan * k
        eff_search = scan * (1.0 - f) + cfg.t_point * f
        travel = cfg.c_move * k
        t = eff_search + travel

        prev = int(self.last_pos[command])
        moved = prev >= 0 and prev != k
        error = False
        corrections = 0
        if moved and self.rng.random() < cfg.p_slip * f:
            error = True
            corrections = 1
            t += cfg.t_correction

        t += abs(self.rng.normal(0.0, cfg.noise_sd))

        if prev == k:
            self.familiarity[command] = min(cfg.fam_cap, f + cfg.fam_gain)
        else:
            self.familiarity[command] = f * cfg.fam_decay
        self.last_pos[command] = k

        return {"time": t, "error": error, "corrections": corrections, "pos": k}
