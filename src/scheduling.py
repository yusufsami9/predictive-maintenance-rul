import numpy as np
from ortools.sat.python import cp_model


def point_estimate_costs(rul_estimate, horizon, failure_cost=100, waste_cost=1):
    """Treat the predicted RUL as if it were the truth.

    These costs use exactly the same failure/wasted-life definition as
    :func:`evaluate_schedule`.  Deterministic tie-breaking in
    :func:`solve_schedule` handles equal-cost late-service/defer decisions.
    """
    est = np.asarray(rul_estimate, dtype=float)
    t = np.arange(1, horizon + 1)[None, :]
    e = est[:, None]

    cost = np.where(
        t <= e,
        np.rint(e - t) * waste_cost,  # life discarded
        failure_cost,  # serviced too late
    )
    cost = np.hstack([np.zeros((len(est), 1)), cost]).astype(int)  # pad t=0, unused

    defer = np.where(est > horizon, 0, failure_cost).astype(int)
    return cost, defer


def expected_costs(pred, errors, horizon, failure_cost=100, waste_cost=1):
    """Price each slot against the model's whole error distribution.

    Validation errors are d = predicted - true, so for an engine predicted at
    p, every observed d implies a plausible true RUL of p - d. Pushing that
    distribution through the cost function gives the *expected* cost of each
    slot instead of its cost under one optimistic guess, and the optimiser
    works out for itself how much buffer each engine deserves.
    """
    pred = np.asarray(pred, dtype=float)
    errors = np.asarray(errors, dtype=float).ravel()
    if errors.size == 0:
        raise ValueError("errors must contain at least one calibration residual")
    possible = pred[:, None] - errors[None, :]  # (n_engines, n_samples)

    cost = np.zeros((len(pred), horizon + 1))
    for t in range(1, horizon + 1):
        scenario_cost = np.where(
            possible < t,
            failure_cost,
            (possible - t) * waste_cost,
        )
        cost[:, t] = scenario_cost.mean(axis=1)

    defer = np.rint((possible <= horizon).mean(axis=1) * failure_cost).astype(int)
    return np.rint(cost).astype(int), defer


def solve_schedule(cost, defer, horizon, capacity, verbose=False):
    """Exact maintenance schedule via CP-SAT, given a priced cost matrix.

    x[i, t] = 1 when engine i is serviced in period t. Leaving every x[i, t] at
    zero means deferring the engine past the window, priced by defer[i].

    Many schedules tie on the priced objective while differing wildly in what
    they actually cost against the hidden truth -- with an arbitrary tie-break
    the same inputs returned realised costs of 1256 or 1457 run to run. So ties
    are broken deterministically, and in a direction that is operationally
    defensible rather than arbitrary: prefer servicing earlier, and prefer
    servicing at all over deferring. Both leave more slack for the prediction
    error the priced objective cannot see.

    The tie-break is scaled to be strictly dominated by the real objective, so
    it only ever chooses among genuinely equal-cost schedules.
    """
    n = cost.shape[0]
    model = cp_model.CpModel()

    x = {(i, t): model.NewBoolVar(f"x_{i}_{t}") for i in range(n) for t in range(1, horizon + 1)}

    for i in range(n):  # each engine at most one slot
        model.Add(sum(x[i, t] for t in range(1, horizon + 1)) <= 1)

    for t in range(1, horizon + 1):  # the shop cannot be oversubscribed
        model.Add(sum(x[i, t] for i in range(n)) <= capacity)

    terms, tie_break = [], []
    for i in range(n):
        deferred = 1 - sum(x[i, t] for t in range(1, horizon + 1))
        for i_t in range(1, horizon + 1):
            terms.append(int(cost[i, i_t]) * x[i, i_t])
            tie_break.append(i_t * x[i, i_t])  # earlier slots preferred
        # (1 - sum_t x[i,t]) equals 1 exactly when engine i is deferred.
        terms.append(int(defer[i]) * deferred)
        tie_break.append((horizon + 1) * deferred)  # worse than any real slot

    # Largest possible tie-break value, so one unit of real cost always outranks
    # the entire tie-break term.
    tie_break_scale = n * (horizon + 1) + 1
    model.Minimize(tie_break_scale * sum(terms) + sum(tie_break))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    # Single worker + fixed seed: multi-threaded search picks whichever optimum
    # it reaches first, which is exactly the non-determinism we are removing.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no solution: {solver.StatusName(status)}")

    if verbose:
        # Report the real priced objective, not the tie-break-scaled one solved.
        planned_cost = solver.ObjectiveValue() // tie_break_scale
        print(
            f"  {solver.StatusName(status)}  planned cost={planned_cost:.0f}  "
            f"{solver.WallTime():.2f}s"
        )

    return {
        i: next((t for t in range(1, horizon + 1) if solver.Value(x[i, t])), None)
        for i in range(n)
    }


def fixed_interval_schedule(n, horizon, capacity):
    """Calendar-based preventive maintenance: ignore condition data and work
    through the fleet at a fixed rate. The baseline predictive maintenance is
    meant to beat."""
    return {i: (i // capacity + 1 if i // capacity + 1 <= horizon else None) for i in range(n)}


def greedy_schedule(rul_estimate, horizon, capacity):
    """Most urgent engine first, booked as late as it can safely go so as not
    to waste life. Myopic -- it never revisits an earlier booking when a more
    urgent engine shows up later."""
    remaining = {t: capacity for t in range(1, horizon + 1)}
    schedule = {}

    for i in np.argsort(rul_estimate):
        if rul_estimate[i] > horizon:
            schedule[i] = None
            continue

        latest = int(min(horizon, np.floor(rul_estimate[i])))
        chosen = next((t for t in range(max(latest, 1), 0, -1) if remaining[t] > 0), None)
        if chosen is None:
            chosen = next((t for t in range(1, horizon + 1) if remaining[t] > 0), None)

        schedule[i] = chosen
        if chosen is not None:
            remaining[chosen] -= 1

    return schedule


def threshold_schedule(rul_estimate, threshold, horizon, capacity):
    """Service everything predicted below a cut-off, soonest first, and leave
    the rest. No solver, no cost model, one number to pick."""
    remaining = {t: capacity for t in range(1, horizon + 1)}
    schedule = {}

    for i in np.argsort(rul_estimate):
        if rul_estimate[i] > threshold:
            schedule[i] = None
            continue
        chosen = next((t for t in range(1, horizon + 1) if remaining[t] > 0), None)
        schedule[i] = chosen
        if chosen is not None:
            remaining[chosen] -= 1

    return schedule


def evaluate_schedule(schedule, rul_actual, horizon, failure_cost=100, waste_cost=1):
    """Score a schedule against the RUL that actually happened.

    schedule maps engine index -> service period, or None if deferred past the
    window. Never sees any prediction -- every strategy is scored on the same
    real outcome.
    """
    failures, wasted = 0, 0
    for i, period in schedule.items():
        actual = rul_actual[i]
        if period is None:
            if actual <= horizon:
                failures += 1  # left in service and it broke
        elif period > actual:
            failures += 1  # the shop slot came too late
        else:
            wasted += actual - period

    return {
        "failures": failures,
        "wasted_life": int(round(wasted)),
        "total_cost": failures * failure_cost + int(round(wasted)) * waste_cost,
    }
