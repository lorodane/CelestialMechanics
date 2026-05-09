# Stable Islands Boundary Experiment

## Goal
This experiment studies how to detect the outer boundary of a stable island in the Sitnikov Poincare section using finite-time non-escape tests.

The main practical objective is to estimate a boundary point from a trusted interior seed and then build a smooth closed boundary curve from iterates near that boundary.

## Implemented Method
Main implementation:
- [../../src/analysis/stable_islands.py](../../src/analysis/stable_islands.py)

Key function:
- outer_irc_point_stable_island

Boundary-search idea:
1. Fix t and move along a ray in v from an interior point p0.
2. Classify a test point as non-escaping if fast_crossings_iterated reaches max_crossings.
3. Increase step size when no expected distance is provided, to quickly find an escape transition.
4. Once escape is detected, step back inward and refine until distance estimate is below dist_error.

Detailed search mechanics used by outer_irc_point_stable_island:
1. Initial step size ds:
	- If expected_dist is None: ds = max(1e-4, dist_error), with auto-expand enabled.
	- If expected_dist is provided: ds = max(expected_dist / 10, dist_error / 10), with no auto-expand.
2. Bracketing phase (when auto-expand is on):
	- While the next test point is still non-escaping, the search advances and doubles ds.
	- This quickly crosses long interior plateaus and reaches the first escape transition.
3. Escape handling and 3-step back:
	- At the first escaping test point v_escape, the algorithm moves back by three steps:
	  v_back_three = v_escape - sign * 3 * ds.
	- Intuition: after one detected escape, moving back by 3*ds gives a conservative inward point
	  that is typically well inside the non-escaping side of the transition.
4. Safety correction for oversized retreat:
	- If v_back_three falls before the current start of the refinement interval, ds is too large.
	- The step is reduced to ds = (v_escape - v_start) / 6 and the local search retries.
5. Refinement scale update:
	- The local distance estimate is distance_estimate = 3 * ds.
	- If distance_estimate < dist_error, the method returns converged with bracket
	  (v_back_three, v_escape).
	- Otherwise, refinement restarts from v_start = v_back_three with
	  ds = max(distance_estimate / 10, dist_error / 10), and auto-expand is disabled.

## Why This Design Was Chosen
The notebooks in this folder showed that finite-time boundary detection is affected by KAM stickiness:
- points near the detected outer boundary can appear irregular, not cleanly quasi-periodic,
- the detected boundary can be slightly outside the true long-time invariant contour.

Because of that, the method separates two concerns:
1. Robustly detect an outer transition from non-escape to escape.
2. Move slightly inward from that detected outer point before generating orbit samples for curve fitting.

This inward pull is empirical but reproducible in the notebook tests and gives better quasi-periodic clouds for spline fitting.

## Evidence From Notebook Development
Main notebook:
- [stable_islands_compute.ipynb](stable_islands_compute.ipynb)

Supporting notebook:
- [compute_stable_islands2.ipynb](compute_stable_islands2.ipynb)

Observed during testing:
- Negative-direction searches can hit the physical floor v = 0 for some seeds.
- Outer boundary trajectories often look KAM-sticky and irregular.
- A small inward interpolation from boundary to seed (for example eps near 0.1 in tested cases) improves quasi-periodic behavior for boundary-cloud generation.

## Practical Workflow
Recommended workflow from one interior seed p0:
1. Call outer_irc_point_stable_island to get an outer boundary estimate.
2. Build an inward-shifted seed between p0 and that boundary point.
3. Iterate phi_fast to generate a cloud of boundary crossings.
4. Fit a periodic closed curve from the cloud.

Convenience function:
- outer_boundary_spline_from_interior_point

This wrapper encodes the same workflow and includes the inward-shift step for KAM-stickiness robustness.

## Parameter Guidance
Boundary detection parameters:
- max_crossings: non-escape horizon (higher is stricter, slower).
- dist_error: target boundary-distance tolerance for refinement.
- max_iterations: outer refinement budget.
- expected_dist:
  - None: auto-expand steps to bracket transition quickly.
  - float: use when a reliable prior boundary distance is known.

Curve fit parameters:
- n_samples: number of orbit crossings used for cloud fitting.
- smoothing, spline_degree, n_eval: trade off smoothness and detail.

## Reproducibility
Run script:
- [run_experiment.py](run_experiment.py)

Metadata output:
- [metadata.json](metadata.json)

When run via the script, timestamp, commit hash, dirty state, and selected parameters are recorded in metadata.json.

## Limitations
- This is a finite-time criterion, not a formal infinite-time invariant-set proof.
- Boundary quality depends on the interior seed quality and non-escape horizon.
- KAM stickiness can bias the raw outer estimate, which is why inward correction is used before orbit-cloud fitting.

