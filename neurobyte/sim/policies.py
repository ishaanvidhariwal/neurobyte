"""Menu-ordering policies.

Each policy exposes ``current_order`` (the menu shown on the next trial) and an
``update(command, trial_idx)`` hook called after every selection.

    FixedOrder          arbitrary fixed layout, never changes         (control)
    StaticFrequency     observe a warm-up prefix, sort once, freeze   (hard baseline)
    AdaptiveOnline      recency-weighted counts, re-sort every trial  (aggressive)
    AdaptiveStabilized  same estimate + hysteresis so items move at
                        most one slot per trial and only when the
                        gain is clear                                 (the proposed method)
"""

from __future__ import annotations

import numpy as np


class Policy:
    name = "policy"

    def __init__(self, n_commands: int):
        self.n = n_commands
        self.current_order: list[int] = list(range(n_commands))

    def update(self, command: int, trial_idx: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class FixedOrder(Policy):
    name = "fixed"

    def update(self, command: int, trial_idx: int) -> None:
        pass


class StaticFrequency(Policy):
    name = "static-freq"

    def __init__(self, n_commands: int, warmup: int):
        super().__init__(n_commands)
        self.warmup = warmup
        self.counts = np.zeros(n_commands)
        self.frozen = False

    def update(self, command: int, trial_idx: int) -> None:
        if self.frozen:
            return
        self.counts[command] += 1.0
        if trial_idx + 1 >= self.warmup:
            self.current_order = [int(i) for i in np.argsort(-self.counts, kind="stable")]
            self.frozen = True


class AdaptiveOnline(Policy):
    name = "adaptive-online"

    def __init__(self, n_commands: int, decay: float = 0.95):
        super().__init__(n_commands)
        self.decay = decay
        self.w = np.zeros(n_commands)

    def update(self, command: int, trial_idx: int) -> None:
        self.w *= self.decay
        self.w[command] += 1.0
        self.current_order = [int(i) for i in np.argsort(-self.w, kind="stable")]


class AdaptiveStabilized(Policy):
    name = "adaptive-stable"

    def __init__(self, n_commands: int, decay: float = 0.95, tau: float = 0.035):
        super().__init__(n_commands)
        self.decay = decay
        self.tau = tau            # min probability-mass advantage needed to move an item up
        self.w = np.zeros(n_commands)

    def update(self, command: int, trial_idx: int) -> None:
        self.w *= self.decay
        self.w[command] += 1.0
        total = self.w.sum()
        if total <= 0.0:
            return
        p = self.w / total
        # One damped bubble pass: swap adjacent entries only when the lower one
        # is wanted more than the upper one by more than tau. An item therefore
        # migrates at most one slot per trial, and noise below tau never moves
        # anything -> the layout stays stable enough to build spatial memory.
        order = list(self.current_order)
        for i in range(len(order) - 1):
            if p[order[i + 1]] - p[order[i]] > self.tau:
                order[i], order[i + 1] = order[i + 1], order[i]
        self.current_order = order
