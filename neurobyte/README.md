# NEUROBYTE — menu-adaptation simulation

A self-contained simulation study of the central claim in the NEUROBYTE
proposal: that a system which *learns a user's interaction pattern from
short-term telemetry and adapts its menu online* beats a non-adaptive
interface on task efficiency.

It is a **simulation**, not a human study. Every number below is a
consequence of the cost model in [`sim/user.py`](sim/user.py). The value of
the exercise is that it forces the proposal's hand-wavy "12.4 s → 9.8 s"
projection into an explicit model with explicit assumptions, and then checks
which way the result actually falls.

## TL;DR result

**Across every regime tested, a static frequency-ordered menu is the fastest
policy. Online adaptation is consistently slower**, by +0.10 s / selection in
a 12-command menu (dz ≈ 2.7) and +0.38 s in a 40-command evolving menu
(dz ≈ 7.7). Stabilising the adapter (hysteresis, one-slot-per-trial moves)
closes about a third of the gap but never overturns it.

This matches the HCI literature: Findlater & McGrenere (CHI 2004) found a
static menu beat an adaptive menu in a controlled study, and frequency-based
static ordering is the standard hard-to-beat baseline. **The proposal's
projected 21 % win is not merely optimistic — it points the wrong way.**

Why the model behaves this way:

1. **Reordering destroys spatial memory.** A stable layout lets the user
   memorise where things are; selection time then collapses toward a
   pointing-time floor regardless of list position. An adaptive layout keeps
   moving items, so that memory never consolidates and every selection pays
   close to full visual-search cost.
2. **Relocation slips.** When a remembered item moves, the user sometimes
   clicks its old location and pays a correction penalty. Curiously the
   *stabilised* adapter has a higher slip rate than the aggressive one
   (fig `fig_walk_summary_bars.png`): it is stable enough for the user to
   build the spatial memory that then gets violated when it does move an item.

The sensitivity sweeps (`fig_sensitivity.png`) show the static-frequency
advantage *shrinks* as the world gets more non-stationary (faster drift, more
churn) but does not close within any plausible range.

## What's modelled

**Demand** (`DemandProcess`) — a seeded, policy-independent stream of which
command the user wants each trial. Popularity is Zipfian. Three drift modes:
`none`, `step` (one abrupt re-permutation of popularity), `walk` (popularity
random-walks a few adjacent swaps per session and never settles).

**Selection cost** (`SelectionCostModel`) — time to pick a command given the
current menu order:

- linear visual scan from the top (cost grows with the target's position);
- spatial memory: stable repeats at the same position drive search time down
  toward a floor `t_point`, capped at `fam_cap < 1` (search never fully
  vanishes);
- relocation slips: if an item moved since last use, slip probability
  `p_slip · familiarity`, costing `t_correction`.

Parameters are in `CostConfig` / `DemandConfig`. They are plausible, not
calibrated to real people — see *Limitations*.

## Policies (`sim/policies.py`)

| policy | behaviour |
|---|---|
| `FixedOrder` | arbitrary fixed layout, never changes (control) |
| `StaticFrequency` | observe a 60-trial warm-up, sort by frequency once, freeze (hard baseline) |
| `AdaptiveOnline` | recency-weighted counts, full re-sort every trial (aggressive) |
| `AdaptiveStabilized` | same estimate + hysteresis `tau`; an item moves at most one slot per trial and only when the probability-mass gain exceeds `tau` |

## Experiments

- **Monte-Carlo comparison** — 200 simulated users; each user's identical
  demand sequence is run through all four policies (paired). Report bootstrap
  95 % CIs and paired differences vs the static-frequency baseline
  (Cohen's dz).
- **Sensitivity sweeps** — mean selection time vs step-drift strength, vs
  slip probability, vs walk churn rate. This is the actual scientific content:
  *which regime favours which policy*.
- **ABAB reversal** — a single adaptive-stabilized policy with adaptation
  toggled on (B) / frozen (A) across session blocks, drift running throughout.
  Isolates that the layout churn itself — not drift, not practice — moves the
  metric. Sessions 9–12 (frozen) reach the run's best performance; 13–16
  (adaptation back on) jump back up.

## Run it

```bash
pip install -r requirements.txt
python run.py            # ~35 s: 200 seeds/scenario + sweeps + ABAB
python run.py --quick    # ~6 s smoke run
```

Outputs to `results/`:

| file | contents |
|---|---|
| `fig_step_learning_curves.png` | per-session time, 12-command menu, one abrupt drift |
| `fig_walk_learning_curves.png` | per-session time, 40-command evolving menu |
| `fig_step_summary_bars.png` / `fig_walk_summary_bars.png` | post-warm-up time / error rate / corrections |
| `fig_sensitivity.png` | the crossover: time vs drift strength, slip prob, churn rate |
| `fig_abab_reversal.png` | ABAB reversal, mean & 95 % band over seeds |
| `summary.json` / `summary.md` | all numbers |

## Limitations (read before quoting any number)

- **Not calibrated to humans.** The cost model encodes qualitative findings
  (linear scan, spatial memory, relocation slips) with invented constants. It
  can show *in which direction and regime* adaptation helps or hurts; it
  cannot predict real millisecond values.
- **One synthetic user model.** No individual differences, no fatigue, no
  learning of the adaptation's logic, no menus-within-menus.
- **Adaptation target is list order only.** Real adaptive UIs also add/hide
  items, resize, or split menus; those are not modelled.
- **The honest takeaway is a negative result.** If you build NEUROBYTE, this
  says: don't pitch "adaptive menu beats static." A defensible positive
  contribution would be *adaptable* (user-controlled) reordering, or adapting
  something other than order, or a spatial-stability-constrained adapter
  evaluated against this baseline — with a real study, honestly powered.

## Files

```
sim/user.py         demand process + selection cost model
sim/policies.py      the four menu-ordering policies
sim/environment.py   run one policy through one demand sequence
sim/experiment.py    Monte-Carlo, bootstrap stats, sweeps, ABAB
run.py               orchestration + figures + summaries
```
