import numpy as np
from ortools.sat.python import cp_model


def point_estimate_costs(rul_estimate, horizon, failure_cost=100, waste_cost=1):
    """Treat the predicted RUL as if it were the truth.

    The late branch grows with lateness so that servicing an overdue engine
    tomorrow still beats abandoning it -- without that term the optimiser is
    indifferent between the two, and it abandons.
    """
    est = np.asarray(rul_estimate, dtype=float)
    t = np.arange(1, horizon + 1)[None, :]
    e = est[:, None]

    cost = np.where(
        t <= e,
        np.rint(e - t) * waste_cost,  # life discarded
        failure_cost + np.rint(t - e) * waste_cost,  # serviced too late
    )
    cost = np.hstack([np.zeros((len(est), 1)), cost]).astype(int)  # pad t=0, unused

    defer = np.where(
        est > horizon, 0, failure_cost + np.rint(horizon + 1 - est) * waste_cost
    ).astype(int)
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
    possible = pred[:, None] - errors[None, :]  # (n_engines, n_samples)

    cost = np.zeros((len(pred), horizon + 1))
    for t in range(1, horizon + 1):
        fails = possible < t
        p_fail = fails.mean(axis=1)
        n_survive = (~fails).sum(axis=1)
        waste_sum = np.where(fails, 0.0, possible - t).sum(axis=1)
        mean_waste = np.where(n_survive > 0, waste_sum / np.maximum(n_survive, 1), 0.0)
        cost[:, t] = p_fail * failure_cost + (1 - p_fail) * mean_waste * waste_cost

    defer = np.rint((possible <= horizon).mean(axis=1) * (failure_cost + 0.5 * horizon)).astype(int)
    return np.rint(cost).astype(int), defer


def solve_schedule(cost, defer, horizon, capacity, verbose=False):
    """Exact maintenance schedule via CP-SAT, given a priced cost matrix.

    x[i, t] = 1 when engine i is serviced in period t. Leaving every x[i, t] at
    zero means deferring the engine past the window, priced by defer[i].
    """
    n = cost.shape[0]
    model = cp_model.CpModel()

    x = {(i, t): model.NewBoolVar(f"x_{i}_{t}") for i in range(n) for t in range(1, horizon + 1)}

    for i in range(n):  # each engine at most one slot
        model.Add(sum(x[i, t] for t in range(1, horizon + 1)) <= 1)

    for t in range(1, horizon + 1):  # the shop cannot be oversubscribed
        model.Add(sum(x[i, t] for i in range(n)) <= capacity)

    terms = []
    for i in range(n):
        for t in range(1, horizon + 1):
            terms.append(int(cost[i, t]) * x[i, t])
        # (1 - sum_t x[i,t]) equals 1 exactly when engine i is deferred.
        terms.append(int(defer[i]) * (1 - sum(x[i, t] for t in range(1, horizon + 1))))

    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no solution: {solver.StatusName(status)}")

    if verbose:
        print(
            f"  {solver.StatusName(status)}  planned cost={solver.ObjectiveValue():.0f}  "
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
