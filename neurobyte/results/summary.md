# NEUROBYTE simulation results

200 simulated users per scenario; 60 per sweep point. All numbers are properties of the cost model in `sim/user.py`, not measurements of human behaviour.

## Scenario `step` (n_commands=12, drift_mode=step)

| policy | time (s) | 95% CI | Δ vs baseline (s) | Cohen's dz | err/trial |
|---|---:|---:|---:|---:|---:|
| Fixed order (control) | 0.780 | [0.774, 0.785] | +0.028 [+0.021, +0.035] | +0.54 | 0.0000 |
| Static frequency (baseline) | 0.752 | [0.747, 0.756] | +0.000 [+0.000, +0.000] | +0.00 | 0.0022 |
| Adaptive online | 0.861 | [0.858, 0.864] | +0.109 [+0.104, +0.114] | +3.02 | 0.0176 |
| Adaptive stabilized | 0.847 | [0.844, 0.850] | +0.095 [+0.090, +0.099] | +2.69 | 0.0221 |

## Scenario `walk` (n_commands=40, drift_mode=walk)

| policy | time (s) | 95% CI | Δ vs baseline (s) | Cohen's dz | err/trial |
|---|---:|---:|---:|---:|---:|
| Fixed order (control) | 1.756 | [1.739, 1.773] | +0.346 [+0.329, +0.364] | +2.75 | 0.0000 |
| Static frequency (baseline) | 1.410 | [1.403, 1.416] | +0.000 [+0.000, +0.000] | +0.00 | 0.0011 |
| Adaptive online | 1.930 | [1.922, 1.939] | +0.520 [+0.512, +0.528] | +9.01 | 0.0127 |
| Adaptive stabilized | 1.788 | [1.781, 1.796] | +0.378 [+0.372, +0.385] | +7.69 | 0.0184 |

## ABAB reversal (adaptive-stabilized only)

- A phases (layout frozen): **0.827 s**
- B phases (adaptation on): **0.816 s** (Δ = -0.011 s)

## Figures

- `fig_step_learning_curves.png`, `fig_step_summary_bars.png`
- `fig_walk_learning_curves.png`, `fig_walk_summary_bars.png`
- `fig_sensitivity.png` — the crossover: which regime favours which policy
- `fig_abab_reversal.png`
