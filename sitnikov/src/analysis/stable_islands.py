from __future__ import annotations

import numpy as np

from src.integrator.integrate import FastSitnikovSimulation
from src.utils.curve_ordering import fit_closed_curve_from_cloud

import warnings


def outer_irc_point_stable_island(
    e,
    p0,
    positive_v_search=True,
    expected_dist=None,
    max_crossings=50,
    dist_error=1e-3,
    max_iterations=10,
):
    """
    Search along a fixed-t ray for the outer boundary point of a stable island.

    A point is classified as non-escaping when
    `sim.crossings_fast(v, t0, max_crossings=max_crossings) == max_crossings`.

    Parameters
    ----------
    e : float
        Sitnikov eccentricity.
    p0 : tuple[float, float]
        Interior point `(v, t)` in section coordinates.
    positive_v_search : bool
        If True, search outward by increasing `v`.
        If False, search outward by decreasing `v` with hard floor at `v=0`.
    expected_dist : float | None
        User estimate of boundary distance from `p0` along the chosen direction.
        If None, the routine auto-brackets by exponentially increasing step size.
    max_crossings : int
        Crossing threshold for non-escape classification.
    dist_error : float
        Target bracket width for boundary localization.
    max_iterations : int
        Maximum number of search/refinement iterations before stopping.

    Returns
    -------
    dict
        {
            "point": (v_hat, t0_mod),
            "distance_estimate": float,
            "status": str,
            "bracket": (v_in, v_out) | None,
        }
    """
    if max_crossings < 1:
        raise ValueError("max_crossings must be >= 1")
    if dist_error <= 0.0:
        raise ValueError("dist_error must be > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    v0, t0 = float(p0[0]), float(p0[1])
    if v0 < 0.0:
        raise ValueError("p0 velocity must satisfy v >= 0")

    t0_mod = float(np.mod(t0, 2.0 * np.pi))
    sign = 1.0 if positive_v_search else -1.0

    sim = FastSitnikovSimulation(e=float(e), solver_method="RK45")

    def is_non_escaping(v_test: float) -> bool:
        if v_test < 0.0:
            return False
        n_cross = sim.crossings_fast(v=v_test, t=t0_mod, max_crossings=int(max_crossings))
        return n_cross == int(max_crossings)

    if not is_non_escaping(v0):
        raise ValueError("Initial point p0 is not classified as non-escaping. Please provide a valid interior point.")

    if expected_dist is not None and expected_dist <= 0.0:
        raise ValueError("expected_dist must be > 0 when provided")

    if expected_dist is None:
        ds = max(1e-4, dist_error)
        auto_expand = True
    else:
        ds = max(float(expected_dist) / 10.0, dist_error / 10.0)
        auto_expand = False


    # Search logic starts here

    v_start = v0

    for _ in range(max_iterations):
        v_curr = v_start

        # Refining loop. Stops when escape detected
        while True:
            v_next = v_curr + sign * ds
            if v_next < 0.0:
                raise ValueError("Search went below v=0, which is not allowed. Check the search direction and expected_dist.")

            if is_non_escaping(v_next):
                v_curr = v_next
                if auto_expand:
                    ds *= 2.0
                continue

            v_escape = v_next
            v_back_three = v_escape - sign * 3.0 * ds
            if v_back_three < v_start:
                ds = (v_escape - v_start) / 6.0
                continue

            distance_estimate = 3.0 * ds
            if distance_estimate < dist_error:
                return {
                    "point": (v_back_three, t0_mod),
                    "distance_estimate": distance_estimate,
                    "status": "converged",
                    "bracket": (v_back_three, v_escape),
                }

            v_start = v_back_three
            ds = max(distance_estimate / 10.0, dist_error / 10.0)
            if auto_expand:
                auto_expand = False
            break


    warnings.warn("Maximum iterations reached without convergence. Returning last best estimate.")
    return {
        "point": (v_start, t0_mod),
        "distance_estimate": np.nan,
        "status": "max_iterations_reached",
        "bracket": None,
    }


def outer_boundary_spline_from_interior_point(
    e,
    p0,
    *,
    n_samples=30,
    positive_v_search=True,
    expected_dist=None,
    max_crossings=50,
    dist_error=1e-3,
    max_iterations=20,
    n_starts=None,
    max_passes=20,
    spline_degree=3,
    smoothing=1e-2,
    n_eval=1000,
):
    """
    Build a smooth outer-boundary spline for a stable island from one interior seed.

    Workflow (matching stable_islands_compute.ipynb):
    1) Compute one outer boundary point from `p0` using `outer_irc_point_stable_island`.
    2) Iterate `phi_fast` from that boundary point to generate a cloud of crossings.
    3) Fit a closed spline through that cloud.

    Parameters
    ----------
    e : float
        Sitnikov eccentricity.
    p0 : tuple[float, float]
        Interior seed point `(v, t)`.
    n_samples : int
        Number of crossings generated by iterating `phi_fast` from boundary seed.
    positive_v_search : bool
        Search direction for all ray boundary probes.
    expected_dist : float | None
        Optional expected boundary distance passed to each ray probe.
    max_crossings, dist_error, max_iterations
        Parameters forwarded to `outer_irc_point_stable_island`.
    n_starts, max_passes, spline_degree, smoothing, n_eval
        Parameters forwarded to `fit_closed_curve_from_cloud`.

    Returns
    -------
    tuple
        (fit, spline), where `fit` is the full payload from
        `fit_closed_curve_from_cloud` plus diagnostics and `spline` is the
        fitted periodic spline object.
    """
    if n_samples < 4:
        raise ValueError("n_samples must be >= 4 to fit a closed boundary reliably")

    initial_boundary_result = outer_irc_point_stable_island(
        e=e,
        p0=p0,
        positive_v_search=positive_v_search,
        expected_dist=expected_dist,
        max_crossings=max_crossings,
        dist_error=dist_error,
        max_iterations=max_iterations,
    )

    # The point returned by `outer_irc_point_stable_island` is affected by KAM stickiness
    # So we pull the seed inward to have a quasi-periodic orbit
    boundary_seed = (
        0.9 * float(initial_boundary_result["point"][0]) + 0.1 * float(p0[0]),
        0.9 * float(initial_boundary_result["point"][1]) + 0.1 * float(p0[1]),
    )

    sim = FastSitnikovSimulation(e=float(e), solver_method="RK45")

    boundary_orbit = np.zeros((int(n_samples), 2), dtype=float)
    p_curr = (float(boundary_seed[0]), float(boundary_seed[1]))
    for i in range(int(n_samples)):
        p_curr = sim.phi_fast(v=float(p_curr[0]), t=float(p_curr[1]))
        boundary_orbit[i, 0] = float(p_curr[0])
        boundary_orbit[i, 1] = float(p_curr[1])

    t_boundary = boundary_orbit[:, 1]
    v_boundary = boundary_orbit[:, 0]

    spline = fit_closed_curve_from_cloud(
        t_boundary,
        v_boundary,
        n_starts=n_starts,
        max_passes=max_passes,
        spline_degree=spline_degree,
        smoothing=smoothing,
        n_eval=n_eval,
    )

    fit = {
        "initial_boundary_result": initial_boundary_result,
        "boundary_seed": (float(boundary_seed[0]), float(boundary_seed[1])),
        "boundary_orbit": boundary_orbit,
        "n_samples": int(n_samples),
        "spline": spline,
    }
    return fit, spline
