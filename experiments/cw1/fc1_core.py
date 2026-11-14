"""FC1 exact finite-clock likelihood, constrained EM and fixed numerical refinement.

No work runs on import. fit_counts is a scientific fitting entry point intended
only for the explicitly authorized Colab fit stage, never author-side tests.
"""
from __future__ import annotations

import math
import platform
from dataclasses import dataclass

import numpy as np


TIMES = np.arange(80, 96, dtype=np.int64)
QUERIES = np.arange(96, 104, dtype=np.int64)
CLOCKS = np.array([(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)], dtype=np.int64)
PARAMETER_NAMES = ("p2", "wH", "wL", "rH", "rL", "rS", "rN", "rho")
FLOOR = 0.01
QUALIFICATION = 1e-6


class ZeroLikelihoodError(ValueError):
    """A probability-valid parameter gives zero mass to an observed history."""


_OPPORTUNITIES = TIMES[None, :] % CLOCKS[:, :1] == CLOCKS[:, 1:]
_OWN_MASKS = np.repeat(_OPPORTUNITIES[:, None, :], 4, axis=1)
for _c in range(6):
    _indices = np.flatnonzero(_OPPORTUNITIES[_c])
    for _b in (0, 1):
        _OWN_MASKS[_c, 2 + _b] = False
        _OWN_MASKS[_c, 2 + _b, _indices[_b::2]] = True
_OWN_N = _OWN_MASKS.sum(axis=2).astype(float)
_NEIGHBOR_N = _OPPORTUNITIES.sum(axis=1).astype(float)
_TARGETS = (QUERIES[None, :] % CLOCKS[:, :1] == CLOCKS[:, 1:]).astype(float)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _theta_array(theta):
    t = np.asarray(theta, dtype=np.float64)
    _require(t.shape == (8,) and np.isfinite(t).all(), "Expected eight finite parameters")
    return t


def validate_theta(theta, tolerance=1e-10):
    """Validate the frozen study's closed feasible region, without changing it."""
    t = _theta_array(theta)
    w = np.array([t[1], t[2], 1 - t[1] - t[2]])
    _require(FLOOR - tolerance <= t[0] <= 1 - FLOOR + tolerance, "Period mass outside bounds")
    _require(np.all(w >= FLOOR - tolerance), "Recording mixture outside floor simplex")
    _require(np.all(t[3:7] >= FLOOR - tolerance) and np.all(t[3:7] <= 1 - FLOOR + tolerance),
             "Retention rate outside bounds")
    _require(t[3] >= t[4] - tolerance, "H/L component order violated")
    _require(-tolerance <= t[7] <= 1 + tolerance, "Sharing probability outside bounds")
    return t


def project(theta):
    """Euclidean projection in the declared eight parameter coordinates.

    The two explicit mixture coordinates have a triangular feasible region;
    this is NOT a three-coordinate simplex projection with a different norm.
    """
    v = _theta_array(theta)
    z = v.copy()
    z[0] = np.clip(v[0], FLOOR, 1 - FLOOR)
    u = v[1:3]
    clipped = np.maximum(u, FLOOR)
    if clipped.sum() > 1 - FLOOR:
        lo, hi = 0.0, max(float(np.max(u) - FLOOR), 0.0) + 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.maximum(u - mid, FLOOR).sum() > 1 - FLOOR:
                lo = mid
            else:
                hi = mid
        clipped = np.maximum(u - hi, FLOOR)
    z[1:3] = clipped
    z[3:7] = np.clip(v[3:7], FLOOR, 1 - FLOOR)
    if z[3] < z[4]:
        z[3] = z[4] = np.clip((v[3] + v[4]) / 2, FLOOR, 1 - FLOOR)
    z[7] = np.clip(v[7], 0, 1)
    validate_theta(z)
    return z


def _histories(x, y):
    x, y = np.asarray(x), np.asarray(y)
    _require(x.ndim == 2 and x.shape[1] == 16 and x.shape == y.shape and len(x) > 0,
             "Histories must be nonempty aligned N by 16 arrays")
    _require(np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all(), "Histories must be Boolean")
    return x.astype(bool), y.astype(bool)


@dataclass
class _Prepared:
    x: np.ndarray
    y: np.ndarray
    counts: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    valid_x: np.ndarray
    valid_y: np.ndarray


def _prepare(x, y, counts=None):
    x, y = _histories(x, y)
    count = np.ones(len(x), dtype=float) if counts is None else np.asarray(counts, dtype=float)
    _require(count.shape == (len(x),) and np.isfinite(count).all() and np.all(count > 0),
             "Counts must be aligned finite positive weights")
    valid_x = ~np.any(x[:, None, None, :] & ~_OWN_MASKS[None, :, :, :], axis=-1)
    valid_y = ~np.any(y[:, None, :] & ~_OPPORTUNITIES[None, :, :], axis=-1)
    return _Prepared(x, y, count, x.sum(axis=1).astype(float), y.sum(axis=1).astype(float), valid_x, valid_y)


def _probabilities(theta, data):
    t = _theta_array(theta)
    p2, wH, wL, rH, rL, rS, rN, rho = t
    wS = 1 - wH - wL
    # A solver may evaluate a point that violates the arbitrary H/L labeling
    # order, but it may never evaluate an invalid probability distribution.
    _require(0 < p2 < 1 and min(wH, wL, wS) > 0 and np.all((t[3:7] > 0) & (t[3:7] < 1))
             and 0 <= rho <= 1, "Invalid probability parameters in objective")
    pi = np.array([p2 / 2, p2 / 2] + [(1 - p2) / 4] * 4)
    weights = np.array([wH, wL, wS / 2, wS / 2])
    rates = np.array([rH, rL, rS, rS])
    kx = data.kx[:, None, None]
    lx = np.exp(kx * np.log(rates)[None, None, :] +
                (_OWN_N[None, :, :] - kx) * np.log1p(-rates)[None, None, :]) * data.valid_x
    ky = data.ky[:, None]
    b = np.exp(ky * math.log(rN) + (_NEIGHBOR_N[None, :] - ky) * math.log1p(-rN)) * data.valid_y
    a = np.sum(lx * weights[None, None, :], axis=2)
    u, v = a @ pi, b @ pi
    shared = (a * b) @ pi
    probability = rho * shared + (1 - rho) * u * v
    _require(np.isfinite(probability).all() and np.all(probability >= 0),
             "An observed pair probability is invalid or nonfinite")
    if np.any(probability == 0):
        raise ZeroLikelihoodError("An observed pair is impossible under the permitted model")
    return dict(pi=pi, weights=weights, rates=rates, lx=lx, a=a, b=b, u=u, v=v,
                shared=shared, probability=probability)


def _score_jacobian(theta, data, parts=None):
    """Derivative of each pair's log probability, in eight raw coordinates."""
    t = _theta_array(theta)
    d = _probabilities(t, data) if parts is None else parts
    pi, a, b, u, v, lx = [d[k] for k in ("pi", "a", "b", "u", "v", "lx")]
    rho = t[7]
    dpi = np.array([.5, .5, -.25, -.25, -.25, -.25])
    out = np.empty((len(data.x), 8), dtype=float)
    out[:, 0] = rho * ((a * b) @ dpi) + (1 - rho) * ((a @ dpi) * v + u * (b @ dpi))
    dlogx = data.kx[:, None, None] / d["rates"][None, None, :] - (
        _OWN_N[None, :, :] - data.kx[:, None, None]) / (1 - d["rates"])[None, None, :]
    dlx = lx * dlogx
    da = [lx[:, :, 0] - .5 * (lx[:, :, 2] + lx[:, :, 3]),
          lx[:, :, 1] - .5 * (lx[:, :, 2] + lx[:, :, 3]),
          t[1] * dlx[:, :, 0], t[2] * dlx[:, :, 1],
          (1 - t[1] - t[2]) * .5 * (dlx[:, :, 2] + dlx[:, :, 3])]
    for j, derivative in enumerate(da, 1):
        out[:, j] = rho * ((derivative * b) @ pi) + (1 - rho) * (derivative @ pi) * v
    db = b * (data.ky[:, None] / t[6] - (_NEIGHBOR_N[None, :] - data.ky[:, None]) / (1 - t[6]))
    out[:, 6] = rho * ((a * db) @ pi) + (1 - rho) * u * (db @ pi)
    out[:, 7] = d["shared"] - u * v
    out /= d["probability"][:, None]
    _require(np.isfinite(out).all(), "Nonfinite likelihood derivative")
    return out


def _objective(theta, data):
    d = _probabilities(theta, data)
    normalized = data.counts / data.counts.sum()
    return float(-normalized @ np.log(d["probability"])), -normalized @ _score_jacobian(theta, data, d)


def objective(theta, x, y, counts=None):
    """Return mean negative log likelihood and its analytic eight-vector gradient."""
    return _objective(theta, _prepare(x, y, counts))


def stationarity(theta, x, y, counts=None):
    value, gradient = objective(theta, x, y, counts)
    residual = float(np.max(np.abs(np.asarray(theta) - project(np.asarray(theta) - gradient))))
    return {"mean_nll": value, "residual": residual, "qualified": residual <= QUALIFICATION,
            "gradient_mean_nll": gradient.tolist(), "threshold": QUALIFICATION}


def _floor_multinomial(counts):
    """Exact MLE for sum(w)=1 and w>=.01, using active-set water filling."""
    n = np.asarray(counts, dtype=float)
    _require(n.shape == (3,) and np.all(n >= 0) and n.sum() > 0, "Invalid component counts")
    active = np.zeros(3, dtype=bool)
    w = np.empty(3, dtype=float)
    for _ in range(4):
        w[active] = FLOOR
        free = ~active
        w[free] = n[free] / n[free].sum() * (1 - FLOOR * active.sum())
        new = free & (w < FLOOR)
        if not new.any():
            return w
        active |= new
    raise ValueError("Mixture active-set update did not terminate")


def _ordered_rates(success, opportunities, old):
    s, n = np.asarray(success, float), np.asarray(opportunities, float)
    zero = n <= 0
    if zero.all():
        return np.asarray(old, float).copy(), [0, 1]
    if zero[0]:
        low = np.clip(s[1] / n[1], FLOOR, 1 - FLOOR)
        return np.array([max(old[0], low), low]), [0]
    if zero[1]:
        high = np.clip(s[0] / n[0], FLOOR, 1 - FLOOR)
        return np.array([high, min(old[1], high)]), [1]
    rates = np.clip(s / n, FLOOR, 1 - FLOOR)
    if rates[0] < rates[1]:
        rates[:] = np.clip(s.sum() / n.sum(), FLOOR, 1 - FLOOR)
    return rates, []


def _em_update(theta, data):
    t = validate_theta(theta)
    d = _probabilities(t, data)
    pi, a, b, u, v, probability = [d[k] for k in ("pi", "a", "b", "u", "v", "probability")]
    nweight = data.counts
    total = float(nweight.sum())
    factor = t[7] * b + (1 - t[7]) * v[:, None]
    own_components = d["lx"] * d["weights"][None, None, :] * pi[None, :, None]
    own_components *= factor[:, :, None] / probability[:, None, None]
    candidate = own_components.sum(axis=2)
    shared = t[7] * a * b * pi[None, :] / probability[:, None]
    independent_neighbor = (1 - t[7]) * u[:, None] * b * pi[None, :] / probability[:, None]
    neighbor = shared + independent_neighbor
    _require(np.max(np.abs(candidate.sum(axis=1) - 1)) < 1e-10 and
             np.max(np.abs(neighbor.sum(axis=1) - 1)) < 1e-10, "E-step normalization failed")
    # The shared branch draws one clock; the independent branch draws two.
    prior = (nweight[:, None] * (candidate + independent_neighbor)).sum(axis=0)
    component = (nweight[:, None, None] * own_components).sum(axis=(0, 1))
    successes = (nweight[:, None, None] * own_components * data.kx[:, None, None]).sum(axis=(0, 1))
    opportunities = (nweight[:, None, None] * own_components * _OWN_N[None, :, :]).sum(axis=(0, 1))
    s = np.array([successes[0], successes[1], successes[2:].sum()])
    n = np.array([opportunities[0], opportunities[1], opportunities[2:].sum()])
    updated = t.copy()
    updated[0] = np.clip(prior[:2].sum() / prior.sum(), FLOOR, 1 - FLOOR)
    weights = _floor_multinomial([component[0], component[1], component[2:].sum()])
    updated[1:3] = weights[:2]
    updated[3:5], zeros = _ordered_rates(s[:2], n[:2], t[3:5])
    zero_names = ["rH" if i == 0 else "rL" for i in zeros]
    if n[2] > 0:
        updated[5] = np.clip(s[2] / n[2], FLOOR, 1 - FLOOR)
    else:
        zero_names.append("rS")
    neighbor_s = float(nweight @ data.ky)
    neighbor_n = float(np.sum(nweight[:, None] * neighbor * _NEIGHBOR_N[None, :]))
    if neighbor_n > 0:
        updated[6] = np.clip(neighbor_s / neighbor_n, FLOOR, 1 - FLOOR)
    else:
        zero_names.append("rN")
    shared_count = float(np.sum(nweight[:, None] * shared))
    updated[7] = np.clip(shared_count / total, 0, 1)
    validate_theta(updated)
    audit = {"expected_clock_draws": float(prior.sum()), "pair_weight": total,
             "expected_shared_pairs": shared_count, "component_counts": [float(component[0]), float(component[1]), float(component[2:].sum())],
             "own_record_successes": s.tolist(), "own_eligible_opportunities": n.tolist(),
             "neighbor_record_successes": neighbor_s, "neighbor_opportunities": neighbor_n,
             "zero_denominator_retained_previous": zero_names}
    return updated, audit


def em_update(theta, x, y, counts=None):
    """One exact constrained EM update; this is not a fit driver."""
    return _em_update(theta, _prepare(x, y, counts))


def _endpoint(theta, data, stage, start_index, extra=None):
    theta = validate_theta(theta)
    value, gradient = _objective(theta, data)
    residual = float(np.max(np.abs(theta - project(theta - gradient))))
    result = dict(theta=theta.tolist(), mean_nll=value, nll_mean=value, total_nll=value * float(data.counts.sum()),
                  feasible=True, qualified=residual <= QUALIFICATION, residual=residual,
                  solverstage=stage, start_index=start_index, gradient_mean_nll=gradient.tolist())
    if extra:
        result.update(extra)
    return result


def fit_counts(x, y, spec):
    """Fit only unlabeled Boolean history pairs with the locked finite budget.

    Every feasible stage endpoint competes by calibration likelihood, including
    unqualified endpoints. No test label, hidden clock, true law or control label
    is accepted by this interface. No fitting executes on module import.
    """
    from scipy.optimize import minimize
    import scipy

    opt = spec.get("optimizer", spec)
    starts = opt["starts"]
    _require(len(starts) == 6, "The frozen design requires six fixed starts")
    for start in starts:
        validate_theta(start)
        _require(0 < start[7] < 1, "An exact sharing-boundary EM start is forbidden")
    x, y = _histories(x, y)
    pairs, counts = np.unique(np.concatenate([x, y], axis=1), axis=0, return_counts=True)
    data = _prepare(pairs[:, :16], pairs[:, 16:], counts)
    endpoints, traces = [], []
    em_max = int(opt.get("em_maxiter", 300))
    slsqp_max = int(opt.get("slsqp_maxiter", 500))
    continuation_max = int(opt.get("continuation_maxiter", 1000))
    ftol = float(opt.get("ftol", 1e-12))
    _require((em_max, slsqp_max, continuation_max, ftol) == (300, 500, 1000, 1e-12), "Optimizer budget differs from frozen contract")
    _require(float(opt.get("stationarity_tolerance", QUALIFICATION)) == QUALIFICATION and
             float(opt.get("em_parameter_tolerance", 1e-9)) == 1e-9 and
             float(opt.get("em_nll_tolerance", 1e-12)) == 1e-12,
             "Optimizer tolerances differ from frozen contract")
    _require(float(opt.get("em_monotonic_tolerance", 1e-10)) == 1e-10 and
             opt.get("rate_bounds", [.01, .99]) == [.01, .99] and
             opt.get("period_mass_bounds", [.01, .99]) == [.01, .99] and
             float(opt.get("component_mass_floor", .01)) == .01 and
             opt.get("sharing_bounds", [0, 1]) == [0, 1] and
             bool(opt.get("ordered_iid_rates", True)), "Parameter domain differs from frozen contract")
    bounds = [(FLOOR, 1 - FLOOR), (FLOOR, 1 - 2 * FLOOR), (FLOOR, 1 - 2 * FLOOR)] + [(FLOOR, 1 - FLOOR)] * 4 + [(0., 1.)]
    constraints = [
        {"type": "ineq", "fun": lambda t: 1 - FLOOR - t[1] - t[2],
         "jac": lambda t: np.array([0., -1., -1., 0., 0., 0., 0., 0.])},
        {"type": "ineq", "fun": lambda t: t[3] - t[4],
         "jac": lambda t: np.array([0., 0., 0., 1., -1., 0., 0., 0.])},
    ]
    for start_index, start in enumerate(starts):
        theta = np.asarray(start, dtype=float)
        previous, _ = _objective(theta, data)
        trace = {"start_index": start_index, "initial_theta": theta.tolist(), "em_mean_nll": [previous], "em_updates": []}
        termination = "iteration_cap"
        for iteration in range(em_max):
            new, audit = _em_update(theta, data)
            current, _ = _objective(new, data)
            _require(current <= previous + 1e-10, "Exact EM likelihood monotonicity failed; preserve attempt and inspect")
            step = float(np.max(np.abs(new - theta)))
            gain = previous - current
            trace["em_mean_nll"].append(current)
            trace["em_updates"].append({"iteration": iteration + 1, "theta": new.tolist(), "max_parameter_step": step,
                                        "mean_nll_gain": gain, "audit": audit})
            theta, previous = new, current
            if step <= 1e-9 or gain <= 1e-12:
                termination = "small_step_or_likelihood_gain_not_a_qualification"
                break
        endpoints.append(_endpoint(theta, data, "em", start_index, {"iterations": iteration + 1, "termination": termination}))
        for stage, budget in (("slsqp", slsqp_max), ("slsqp_continuation", continuation_max)):
            callback_trace = []
            zero_likelihood_trials = []

            def solver_objective(t):
                try:
                    return _objective(t, data)
                except ZeroLikelihoodError:
                    # This is the exact extended-value objective, not a finite
                    # penalty or a hidden cap rho<=1-epsilon. No derivative
                    # exists here: the finite zero sentinel only allows the
                    # solver to reject the infinite trial. It is never used in
                    # an endpoint qualification.
                    zero_likelihood_trials.append(np.asarray(t).tolist())
                    return np.inf, np.zeros(8)

            def callback(t):
                value, _ = solver_objective(t)
                callback_trace.append({"theta": np.asarray(t).tolist(), "mean_nll": value if np.isfinite(value) else None,
                                       "finite_likelihood": bool(np.isfinite(value))})

            answer = minimize(solver_objective, theta, jac=True, method="SLSQP", bounds=bounds,
                              constraints=constraints, callback=callback,
                              options={"maxiter": budget, "ftol": ftol, "disp": False})
            raw = np.asarray(answer.x, dtype=float)
            feasible = project(raw)
            projection_step = float(np.max(np.abs(feasible - raw)))
            # Projection is explicitly saved and re-evaluated, never treated as
            # an untouched optimizer endpoint or proof of a successful solver.
            metadata = {"iterations": int(answer.nit), "solver_success": bool(answer.success),
                        "solver_status": int(answer.status), "solver_message": str(answer.message),
                        "raw_solver_theta": raw.tolist(), "feasibility_projection_step": projection_step,
                        "solver_trace": callback_trace, "zero_likelihood_trial_parameters": zero_likelihood_trials}
            try:
                endpoint = _endpoint(feasible, data, stage, start_index, metadata)
            except ZeroLikelihoodError:
                endpoint = dict(theta=feasible.tolist(), mean_nll=None, nll_mean=None, total_nll=None,
                                feasible=False, parameter_constraints_feasible=True, qualified=False, residual=None,
                                solverstage=stage, start_index=start_index,
                                rejection="Zero likelihood for an observed pair; not eligible for finite-fit selection", **metadata)
            endpoints.append(endpoint)
            # Continue from the best feasible endpoint seen for this start,
            # retaining any better EM point if the numerical solver worsens it.
            best = min((e for e in endpoints if e["start_index"] == start_index and e["feasible"]), key=lambda e: e["mean_nll"])
            theta = np.asarray(best["theta"])
            if best["qualified"]:
                break
        traces.append(trace)
    for i, endpoint in enumerate(endpoints):
        endpoint["endpoint_index"] = i
    chosen = min((i for i in range(len(endpoints)) if endpoints[i]["feasible"]), key=lambda i: (endpoints[i]["mean_nll"], i))
    best = endpoints[chosen]
    return {"selected_theta": best["theta"], "selected_endpoint": chosen, "selected_nll_mean": best["mean_nll"],
            "selected_qualified": best["qualified"], "qualified": best["qualified"],
            "selected_residual": best["residual"], "endpoints": endpoints, "em_traces": traces,
            "n_pairs": int(len(x)), "unique_pair_histories": int(len(pairs)), "n_starts": len(starts),
            "selection": "lowest feasible calibration mean NLL, regardless of qualification; index breaks exact ties",
            "qualification_threshold": QUALIFICATION,
            "qualification_scope": "projected-gradient first-order diagnostic; not global optimality or parameter identifiability",
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}


def _from_posterior(posterior):
    posterior = np.asarray(posterior, dtype=float)
    _require(posterior.ndim == 2 and posterior.shape[1] == 6 and np.isfinite(posterior).all() and
             np.all(posterior >= 0) and np.max(np.abs(posterior.sum(axis=1) - 1)) < 1e-10,
             "Invalid clock posterior")
    masses = np.column_stack([posterior[:, :2].sum(axis=1), posterior[:, 2:].sum(axis=1)])
    choose_two = masses[:, 0] >= masses[:, 1]
    selected = np.where(choose_two[:, None], CLOCKS[:, 0] == 2, CLOCKS[:, 0] == 4)
    hard_posterior = posterior * selected
    hard_posterior /= hard_posterior.sum(axis=1, keepdims=True)
    terms = np.zeros_like(masses)
    np.log2(masses, out=terms, where=masses > 0)
    entropy = -np.sum(masses * terms, axis=1)
    soft, hard = posterior @ _TARGETS, hard_posterior @ _TARGETS
    _require(np.all((soft >= -1e-12) & (soft <= 1 + 1e-12)) and
             np.all((hard >= -1e-12) & (hard <= 1 + 1e-12)), "Forecast outside probability domain")
    # Exact certain-event sums can round to 1+2e-16. Only this bounded
    # roundoff is projected; no substantive probability floor is introduced.
    return {"soft": np.clip(soft, 0, 1), "hard": np.clip(hard, 0, 1),
            "period_masses": masses, "entropy": entropy, "map_period": np.where(choose_two, 2, 4)}


def predict(x, y, theta):
    """Four forecasts from the same fitted population model, no hidden truth."""
    validate_theta(theta)
    data = _prepare(x, y)
    d = _probabilities(theta, data)
    own = d["a"] * d["pi"][None, :]
    own /= own.sum(axis=1, keepdims=True)
    context = d["a"] * d["pi"][None, :] * (theta[7] * d["b"] + (1 - theta[7]) * d["v"][:, None])
    context /= context.sum(axis=1, keepdims=True)
    result = {"posterior_own": own, "posterior_context": context}
    for prefix, posterior in (("own", own), ("context", context)):
        result.update({prefix + "_" + key: value for key, value in _from_posterior(posterior).items()})
    return result


def local_sensitivity(theta, x, y):
    """Observed-history score-Jacobian diagnostic, not an exact identifiability certificate."""
    validate_theta(theta)
    data = _prepare(x, y)
    scores = _score_jacobian(theta, data)
    scaled = scores / math.sqrt(len(scores))
    singular = np.linalg.svd(scaled, compute_uv=False)
    singular = np.pad(singular, (0, max(0, 8 - len(singular))))
    threshold = float(singular[0] * 1e-8) if singular[0] else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return {"score_jacobian_singular_values": singular.tolist(), "relative_rank_cutoff": 1e-8,
            "numerical_rank": rank, "condition_number": condition, "n_history_pairs": len(data.x),
            "coordinate_order": list(PARAMETER_NAMES),
            "scope": "empirical local log-likelihood sensitivity at this theta; finite precision, raw coordinates, observed support only; not global identifiability",
            "boundary_warning": "At active bounds or H=L, unconstrained coordinate rank does not certify feasible-direction identification"}
