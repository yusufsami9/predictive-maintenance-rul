# Predictive Maintenance: RUL Prediction + Maintenance Scheduling

Turbofan engine Remaining Useful Life (RUL) prediction on NASA's C-MAPSS
dataset, extended past model metrics into an optimisation layer that turns
predictions into an actual maintenance schedule.

Most RUL projects stop at "here is my RMSE." This one also asks: given these
predictions, what should a maintenance planner actually do, and does the
answer hold up under scrutiny?

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
turbofan degradation simulation, FD001 subset: single operating condition,
single fault mode. 100 training engines run to failure, 100 test engines
truncated before failure with the true RUL given separately.

Not included in this repo (`data/` is gitignored) — download `train_FD001.txt`,
`test_FD001.txt`, `RUL_FD001.txt` from the link above and place them in `data/`.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Sensor trends, constant-sensor removal, engine lifetime distribution |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | RUL labeling, engine-wise train/val split, rolling features, four models (Linear Regression, Random Forest, XGBoost, LSTM), test-set evaluation |
| [`03_optimization.ipynb`](notebooks/03_optimization.ipynb) | Turns predictions into a maintenance schedule with OR-Tools CP-SAT, bootstrapped |
| [`04_fd004_stress_test.ipynb`](notebooks/04_fd004_stress_test.ipynb) | Ports the pipeline to FD004 (6 operating conditions, 2 fault modes), diagnoses why it breaks, fixes it |

Shared logic (evaluation, sequence windowing, the LSTM, the scheduler) lives
in [`src/`](src/) and is imported by whichever notebooks need it, rather than
redefined in each one — see [Repo layout](#repo-layout).

## Results (held-out test set, 100 engines)

RUL capped at 125 cycles for training, matching the standard C-MAPSS
convention (degradation is weak before that point).

Metrics are against the **125-capped** test RUL, matching the training target
— 11 of the 100 test engines genuinely have more than 125 cycles left, and no
model here can predict above the cap. Uncapped figures are in the notebook and
don't change the ranking (LSTM 14.53 / 385 uncapped vs 13.49 / 360 capped).

| Model | RMSE | NASA score |
|---|---|---|
| Linear Regression | 21.00 | 1326 |
| Random Forest | 18.11 | 1241 |
| XGBoost | 17.82 | 941 |
| **LSTM** | **13.49** | **360** |

The LSTM reads 30-cycle sequences of the 14 raw (non-constant) sensors and
wins on both metrics simultaneously. The tree models got those same raw
sensors *plus* rolling mean/std features — but one cycle at a time, so the
only trend information available to them was whatever those summaries
preserve, and a mean and a standard deviation are identical for a rising and
a falling window. Full writeup of that comparison, including a validation-set
ranking that reverses on test, is in `02_feature_engineering.ipynb`.

The **NASA score** ([Saxena et al. 2008](https://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2008/phmc08_challenge_00.pdf))
penalizes late predictions (engine believed healthier than it is) far more
than early ones — flying past the safe operating window is more dangerous
than retiring an engine early. That asymmetry shows up again in the
optimisation layer's cost model.

## From prediction to decision

A predicted RUL isn't a decision. A planner needs: *which engines go into
the shop this cycle*, given limited capacity. `03_optimization.ipynb` builds
that layer on top of the LSTM's test predictions.

**Setup:** 100 engines, 30-cycle planning horizon, a capacity-constrained
shop, minimizing `wasted remaining life (1/cycle) + unplanned failure (100)`
— the same late-vs-early asymmetry as the NASA score. Solved exactly with
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp) (cross-checked
against Gurobi on one scenario — both solvers land on the identical schedule
and cost).

One thing worth flagging, because it silently corrupted an earlier version of
these numbers: **many schedules tie on the priced objective while differing
substantially in what they actually cost against the hidden truth.** With an
arbitrary tie-break, identical inputs returned realised costs of 1256 or 1457
run to run — all of them provably optimal. The solver now breaks ties
deterministically, preferring earlier service and preferring service over
deferral, both of which leave more slack for the prediction error the priced
objective can't see. Any exact-optimisation result reported without checking
this is reporting one arbitrary member of a tied set.

**Headline finding, and it's the one that survived scrutiny:** accounting for
prediction uncertainty is worth roughly **3x**, whether you do it with a
fixed safety margin, an expected-cost objective over the model's error
distribution, or a simple threshold rule. Scheduling against the raw point
estimate is what's fragile — the exact solver spends every believed cycle of
slack and leaves zero buffer for the ~14-cycle RMSE the model actually has.

```
capacity 3, bootstrapped over 200 resampled fleets (95% interval)
  Greedy + safety margin        429   [309,  576]
  CP-SAT + safety margin        458   [309,  793]
  CP-SAT, expected cost         473   [294,  697]
  Threshold rule (oracle)       476   [278,  701]
  ── uncertainty-aware strategies above, raw point estimates below ──
  Greedy, raw prediction       1477   [762, 2154]
  CP-SAT, raw prediction       1489   [762, 2221]
```

The threshold rule is marked *oracle* because its cut-off is re-tuned on the
test outcome of every replicate. That makes it an upper bound on what the
policy family could do, not a number a deployed threshold would reach — it
is in the table as a deliberately strong rival, not as a deployable
strategy.

**What did *not* survive bootstrapping:** an earlier pass claimed the
expected-cost formulation was clearly the best uncertainty-aware method. It
wasn't — that read came from a single 100-engine sample scored once. Across
200 resampled fleets, the uncertainty-aware strategies are statistically
indistinguishable at this capacity: P(expected cost cheaper) is 0.38 against
greedy+margin and 0.44 against CP-SAT+margin, both coin flips. The one place
exact optimisation demonstrably earns its complexity is under tighter
capacity (1 slot/cycle), where a uniform margin can't express which engines
deserve the scarce early slots and the expected-cost model beats it in 100%
of replicates.

**Does the 100:1 price hold the conclusion up?** The notebook re-optimises
every strategy from scratch at ratios from 20:1 to 500:1, rather than
re-pricing a schedule that was chosen at 100:1 — those answer different
questions, and only the first one is a sensitivity analysis. The
expected-cost strategy visibly adapts (211 at 20:1, 503 at 500:1) while the
raw-prediction strategies degrade steeply as failures get more expensive
(338 → 7538). The uncertainty-aware-beats-raw conclusion holds at every
ratio tested.

A rolling-horizon simulation (re-plan every cycle instead of once) is also
in the notebook, with an explicit caveat: FD001's test trajectories are too
short past truncation for anything to actually fail inside the simulated
window, so that comparison doesn't support a conclusion either way. Left in
because the honest negative result — plus the finding that travels with it,
that prediction RMSE drops from ~21 to ~14 as an engine approaches the
window's end — is more useful than deleting it.

## Stress test: FD004

FD001 has one operating condition and one fault mode — the easy C-MAPSS
subset. FD004 has six operating conditions and two fault modes, and is the
standard harder benchmark. `04_fd004_stress_test.ipynb` asks: does the 02
pipeline still work if ported as-is?

It doesn't, and the reason is diagnosable. The 7 sensors FD001's EDA called
"constant" and dropped aren't constant under six operating regimes — their
apparent variance is mostly which regime the engine is in, not degradation.
Standardizing over the whole dataset without accounting for that:

```
val RMSE 26.57, NASA score 171067
```

Clustering `(op1, op2, op3)` into 6 groups with k-means and z-scoring each
sensor within its own cluster (statistics from the training split only)
recovers most of the gap:

```
val RMSE 16.72, NASA score 93550   (37% RMSE reduction)
```

Held out FD004 test set (237 of 248 engines; 11 too short for a 30-cycle
window, excluded and counted): **RMSE 15.58, NASA score 1458** — 6.15/engine
against FD001's 3.60/engine. FD004 stays harder even after the fix, which is
what "two fault modes instead of one" should look like.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost torch ortools matplotlib
# gurobipy optional, only used for the one cross-check cell
```

**macOS:** XGBoost and PyTorch each bundle their own OpenMP runtime, and
whichever loads second segfaults the kernel. `brew install libomp` plus
`OMP_NUM_THREADS=1` (set at the top of `02_feature_engineering.ipynb`, before
any of these libraries import) avoids the clash at no measurable cost on
data this size.

Notebooks import shared code as `sys.path.append('..')` then `from src...
import ...` — run them from inside `notebooks/` (the default for Jupyter)
and this resolves automatically.

Run notebooks in order: `01` → `02` → `03`, plus `04` independently (it only
needs `01`'s reasoning, not its outputs). `02` exports predictions and
validation errors to `outputs/`, which `03` reads — no retraining needed to
run the optimisation layer.

## Known limitations

Things that are wrong-ish and known, rather than wrong and hidden:

- **The optimiser's cost function and the evaluator's are not identical.**
  The evaluator charges a flat penalty for a failure; the optimiser's cost
  matrix adds a term that grows with lateness. That term exists for a real
  reason — without it the solver is indifferent between servicing an overdue
  engine tomorrow and abandoning it forever, and it abandons — but it means
  "optimal" is optimal for the surrogate, not for the stated economics. The
  clean fix is one cost definition covering early service, late service,
  failure, deferral and post-failure downtime, used by both. That needs
  post-failure economics to be *decided*, which is a modelling call, not a
  code change.
- **The uncertainty distribution is mildly optimistic.** The LSTM's
  best epoch is chosen on validation RMSE, and the residuals exported as the
  scheduler's error distribution come from that same validation set. It isn't
  test leakage, but validation is doing two jobs. There's also a sampling
  mismatch: calibration uses 3490 heavily-overlapping windows from 20
  engines, while scheduling applies one final prediction per engine.
  Engine-level out-of-fold residuals would be the honest version.
- **Single split, single seed.** One engine-wise split and one torch seed
  produced 13.49. Repeated splits or seeds would say how much of that is
  stable.
- **FD004 excludes 11 of 248 test engines** that are shorter than the
  30-cycle window, so its test number is not a complete FD004 benchmark.

## What's next

- **Per-engine uncertainty**: the optimisation layer currently uses one
  fleet-wide error distribution. Quantile regression or MC dropout would let
  the scheduler know when the model itself is unsure about a specific engine.
- **A cost model grounded in something other than an assumption**: the 100:1
  failure/waste ratio and 30-cycle horizon were picked to make the problem
  interesting, not derived from real maintenance economics.
- **FD004 architecture tuning**: 04 reuses FD001's LSTM capacity (hidden
  size 64, 2 layers, 30-cycle window) unchanged. Whether that's still the
  right capacity for 2.5x the engines and two fault modes is untested.

## Repo layout

```
data/            gitignored — see Dataset section
notebooks/       01_eda, 02_feature_engineering, 03_optimization, 04_fd004_stress_test
outputs/         predictions + errors exported by 02, consumed by 03
src/
  data.py        column layout shared by every C-MAPSS subset
  evaluation.py  nasa_score
  sequences.py   create_sequences (per-cycle rows -> fixed-length LSTM windows)
  lstm_model.py  LSTMRegressor, train_lstm (with best-epoch checkpointing)
  scheduling.py  cost models, CP-SAT solver, baselines, schedule scoring
```
