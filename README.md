# Predictive Maintenance: RUL Prediction + Maintenance Scheduling

Remaining Useful Life (RUL) prediction on NASA's simulated C-MAPSS turbofan
data, followed by a capacity-constrained maintenance scheduling experiment.
The project deliberately reports negative and unstable results as well as the
successful ones: prediction accuracy does not automatically make an optimizer
useful, and a single neural-network seed is not a reliable conclusion.

## Dataset

The data come from NASA's [Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
The main experiment uses FD001 (one operating condition and one fault mode):
100 run-to-failure training engines and 100 test engines truncated before
failure. FD004 (six operating conditions and two fault modes) is a separate
stress test.

The data are not committed (`data/` is gitignored). Download the C-MAPSS files
from NASA and place the FD001 and FD004 train, test, and RUL text files in
`data/`. Please acknowledge the dataset source as NASA requests. The asymmetric
metric follows the PHM08 challenge described by
[Saxena et al. (2008)](https://doi.org/10.1109/PHM.2008.4711414).

## Pipeline

| Notebook | Purpose |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Sensor trends, constant-sensor checks, lifetime distribution |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Engine-wise split, features, four predictors, full-data refit, seed stability, cross-fitted calibration |
| [`03_optimization.ipynb`](notebooks/03_optimization.ipynb) | Capacity-constrained schedules, sensitivity analysis, rolling-horizon caveat, paired bootstrap |
| [`04_fd004_stress_test.ipynb`](notebooks/04_fd004_stress_test.ipynb) | FD004 operating-condition diagnosis and complete 248-engine test coverage |

Shared implementations live in [`src/`](src/), and focused regression tests
cover the metric, sequence boundaries, cost consistency, and solver feasibility,
determinism, and primary optimality.

## FD001 prediction results

RUL is capped at 125 cycles as the training target. The primary table therefore
uses capped test truth, but predictions are not forcibly clipped. Uncapped RMSE
and NASA score are also reported in notebook 02.

After selecting the LSTM epoch on a 20-engine validation split, every model is
refit on all 100 training engines. The table uses the predetermined LSTM seed
42; the test set is not used to choose a seed.

| Model | Capped RMSE | Capped NASA | Uncapped RMSE | Uncapped NASA |
|---|---:|---:|---:|---:|
| XGBoost | 17.21 | 823 | 18.20 | 855 |
| Random Forest | 18.07 | 1030 | 19.00 | 1063 |
| Linear Regression | 21.03 | 1337 | 22.02 | 1392 |
| LSTM, seed 42 | 17.30 | 1393 | 18.26 | 1429 |

The LSTM is best on the common 3,490-window validation sample (RMSE 13.87),
but that ranking reverses on test. More importantly, fixed-epoch full refits
are highly seed-sensitive:

| LSTM seed | Capped RMSE | Capped NASA |
|---:|---:|---:|
| 42 | 17.30 | 1393 |
| 43 | 15.73 | 503 |
| 44 | 14.06 | 363 |
| Mean ± sample SD | 15.70 ± 1.62 | 753 ± 559 |

**No model is declared the winner here, and the table is deliberately not
bolded.** Seed 42 was fixed in advance, so quoting it is not cherry-picking —
but it is also the *worst* of the three seeds, and the LSTM's three-seed mean
(15.70) is lower than XGBoost's 17.21. With three seeds the 95% interval on
that mean is roughly [11.7, 19.7], which contains XGBoost. The supported
statement is that **these two cannot be separated on this test set**, not that
either one wins. Reading "XGBoost wins" off seed 42 would be the mirror image
of the earlier mistake of reading "the LSTM wins" off a single lucky run.

Two caveats that keep this comparison from being clean:

- **The seed budget is asymmetric.** The LSTM was measured across three seeds;
  XGBoost and Random Forest were run once each at `random_state=42`. Tree
  ensembles have their own seed variance and it was never measured here, so a
  three-seed model is being compared against a one-seed point estimate.
- **The refit protocol probably costs the LSTM something.** The epoch count was
  selected on 80 engines and then reused verbatim for the 100-engine refit, but
  25% more data at the same epoch count means 25% more gradient steps. Epoch
  count is not a transferable hyperparameter across dataset sizes, so part of
  the 13.49 → 17.30 drop is likely protocol rather than a property of the model.
  Re-selecting the epoch count for the refit size, or keeping the checkpointed
  80-engine model as the deliverable, are both defensible alternatives.

The flat models and LSTM are now compared on exactly the same validation rows:
the first 29 cycles of each engine are excluded for every model because they
cannot form a 30-cycle sequence.

## From prediction to a maintenance decision

Notebook 03 schedules 100 engines over a 30-cycle horizon with three shop slots
per cycle. The illustrative economics charge 100 for an unplanned failure and
1 per discarded cycle of remaining life. The cost matrices and realized
evaluator now share one definition:

- service on or before failure: discarded remaining life;
- service after failure: one flat failure charge;
- defer past the horizon: one failure charge only if failure occurs within it.

OR-Tools CP-SAT solves the discrete model. Its deterministic secondary
tie-break prefers earlier service and service over deferral only among schedules
with the same primary cost.

Uncertainty is calibrated with five-fold cross-fitting over the 80 engines that
did not participate in epoch selection. Each is held out, artificially truncated
once at an independently sampled RUL, and contributes one residual. The model in
that fold is trained on the other 84 engines (the 64 remaining calibration-pool
engines plus the 20 selection engines) for the already-selected fixed epoch
count. This replaces 3,490 overlapping residuals from the epoch-selection set
with 80 engine-level out-of-fold residuals. The regenerated margin and scheduling
results below come from that separated calibration sample; its 90th percentile
is a 26.04-cycle safety margin.

At capacity 3, 200 fleet bootstrap replicates give:

```text
CP-SAT, expected cost           391  [ 237,  627]
Oracle-retuned threshold        453  [ 278,  684]
Greedy + safety margin          579  [ 391,  828]
CP-SAT + safety margin          667  [ 434, 1094]
Greedy, raw prediction         1853  [1213, 2604]
CP-SAT, raw prediction         1895  [1236, 2618]
```

The strong conclusion is narrow: ignoring prediction error is expensive under
these assumed economics. Paired expected-cost-minus-raw intervals exclude zero
by a large margin, in 100% of replicates.

Against the *robust* heuristics the picture is genuinely mixed, and both halves
of it matter — the paired mean difference and how often each side actually wins
say different things, because the differences are heavy-tailed:

| capacity 3, expected cost minus | mean | paired 95% | wins |
|---|---:|---|---:|
| greedy + margin | -188 | [-432, 80] | 92% |
| CP-SAT + margin | -276 | [-646, 22] | 96% |
| oracle threshold | -62 | [-112, -24] | 99% |

Against both margin-based strategies, expected cost is cheaper in the large
majority of resampled fleets, but the interval on the *average* saving still
clips zero — a 92% win rate is real evidence, just not a 95%-level claim about
the mean. Against the oracle threshold at this capacity it is a clean win: the
interval excludes zero and it wins 99% of replicates.

Capacity 1 is where it reverses. Expected cost still beats CP-SAT+margin in
100% of replicates and greedy+margin in 94%, but **loses to the oracle
threshold** — 26% wins, 61% losses, mean +87 [-281, 409]. When slots are
scarcest, an oracle-tuned cutoff is the strongest strategy in the comparison.

Overall: **exact optimization has not conclusively earned its extra operational
complexity.** It is ahead of the deployable heuristics almost everywhere, but
usually with an interval that clips zero, and the one comparator it clearly
loses to at tight capacity is a benchmark nobody could actually deploy.

The threshold comparator is intentionally optimistic: its cutoff is re-tuned
on the observed test outcome in every replicate and at every failure-cost ratio.
It is an oracle benchmark, not a deployable result.

### Sensitivity and solver cross-check

Every strategy is rebuilt at failure costs 20, 50, 100, 200, and 500 rather
than merely repricing one fixed schedule. The oracle-selected threshold changes
from 25 to 40 to 50 across those ratios. All listed error-aware approaches cost
less than raw scheduling at ratios 50–500; at 20, only expected cost and the
oracle threshold do. The conclusion remains conditional on the hypothetical
economics and this one test fleet.

Gurobi and CP-SAT match the primary point-cost objective (1806) in the guarded
cross-check. Their schedules and realized costs do not match: CP-SAT's explicit
tie-break realizes 604, while Gurobi's arbitrary primary-optimal schedule
realizes 2193. This is not solver disagreement about the optimum; it demonstrates
why a specified tie-break matters when the primary objective has many optima.

The rolling-horizon section remains a deliberately inconclusive diagnostic.
The recorded FD001 trajectories do not continue far enough past the simulated
decision point for any engine to fail during the 15-cycle rolling window, so it
cannot support a claim that replanning helps or hurts.

## FD004 stress test

Applying FD001-style global standardization to FD004 confounds degradation with
six operating regimes. On the engine-wise validation split:

| FD004 validation pipeline | RMSE | NASA score |
|---|---:|---:|
| Naive global normalization | 26.57 | 171067 |
| Six-cluster condition-aware normalization | **16.72** | **93550** |

The valid result is that condition-aware normalization substantially improves
validation performance. For the official test population:

- the original 30-cycle model scores RMSE 15.58 / NASA 1458 on 237 engines;
- 11 engines cannot form that window;
- the window is set to the shortest unlabeled test history (19 cycles), its epoch
  count is selected on validation, and it is refit on all 249 training engines;
  it scores capped RMSE **17.37** / NASA **2035** on all **248** test engines.
  Against uncapped truth it scores RMSE 28.64 / NASA 6226.

The 19-cycle result solves the coverage problem but does not prove that 19 cycles
is a better architecture. Cross-subset metrics also do not prove FD004 is harder
in this run: its capped RMSE is slightly higher than FD001 seed 42, while its
NASA score per engine is lower. Dataset size, endpoint distribution, window
length, and FD001 seed instability are confounded.

## Reproduce

The committed notebooks were executed with Python 3.14.0 and the exact package
versions in [`requirements.txt`](requirements.txt).

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook
```

Run notebooks `01` → `02` → `03`. Notebook `04` is a separate FD004 experiment
but reads the regenerated FD001 export for its comparison table. Notebook `02`
writes `test_predictions.csv`, `lstm_val_errors.csv`, and
`rolling_predictions.csv` to `outputs/`; notebook `03` consumes them.

On macOS, XGBoost and PyTorch may load competing OpenMP runtimes. The modeling
notebooks set `OMP_NUM_THREADS=1` before importing either library. Gurobi 13.0.2
is optional and only used by a guarded cross-check cell; OR-Tools is the required
open-source solver.

## Known limitations

- The 125-cycle target cap is a modeling convention. Capped results align with
  training, while uncapped results expose the loss on genuinely longer-lived
  test engines; neither definition is universally correct.
- Three LSTM seeds are enough to expose instability, not enough to estimate its
  full distribution. There is still one engine split and no nested
  hyperparameter-selection protocol.
- Calibration is out-of-fold and engine-level, but it uses one artificial
  truncation per engine and one global residual distribution. Fold models train
  on 84 engines while the final predictor trains on 100.
- The 100:1 cost ratio, 30-cycle horizon, and capacity values are illustrative,
  not derived from real maintenance operations. No schedule should be deployed
  from this study.
- Bootstrap intervals resample the same 100-engine test fleet; they are not an
  external replication. The oracle threshold intentionally uses test outcomes.
- The FD004 study trains a separate model and changes the sequence length for
  full coverage; the unlabeled test history lengths determine that 19-cycle
  choice. It is not a controlled transfer/generalization experiment.
- The rolling-horizon test data are insufficient to evaluate actual failures
  during replanning.

## Repository layout

```text
data/            gitignored NASA C-MAPSS files
notebooks/       four executed analysis notebooks
outputs/         FD001 predictions and cross-fitted residuals consumed by 03
src/
  data.py        shared C-MAPSS column layout
  evaluation.py  asymmetric NASA score
  sequences.py   engine-bounded sequence windows
  lstm_model.py  selected-epoch and fixed-epoch LSTM training
  scheduling.py  cost models, baselines, CP-SAT solver, realized evaluator
tests/           focused unit/regression tests
requirements.txt exact executed environment
LICENSE          MIT license
CITATION.cff     software citation metadata
```
