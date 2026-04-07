import numpy as np
from scipy.interpolate import make_splprep


def _signed_angular_delta(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    """Return signed shortest angular difference a-b in (-period/2, period/2]."""
    return (a - b + 0.5 * period) % period - 0.5 * period


def _pairwise_theta_v_distances(theta: np.ndarray, v: np.ndarray, period: float) -> np.ndarray:
    """Pairwise distances in (theta, v) using circular theta metric."""
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)

    dtheta = _signed_angular_delta(theta[:, None], theta[None, :], period)
    dv = v[:, None] - v[None, :]
    D = np.sqrt(dtheta * dtheta + dv * dv)
    np.fill_diagonal(D, np.inf)
    return D


def _unwrap_angles(theta: np.ndarray, period: float) -> np.ndarray:
    """Unwrap angular samples to a continuous 1D coordinate."""
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return theta.copy()

    steps = _signed_angular_delta(theta[1:], theta[:-1], period)
    unwrapped = np.empty_like(theta)
    unwrapped[0] = theta[0]
    unwrapped[1:] = theta[0] + np.cumsum(steps)
    return unwrapped


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distance matrix for 2D points."""
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    return D


def path_length(D: np.ndarray, order: np.ndarray, closed: bool = True) -> float:
    """Compute total length of path induced by order."""
    order = np.asarray(order, dtype=int)
    length = np.sum(D[order[:-1], order[1:]])
    if closed:
        length += D[order[-1], order[0]]
    return float(length)


def nearest_neighbor_order(
    points: np.ndarray,
    closed: bool = True,
    n_starts: int | None = None,
    distance_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-start greedy nearest-neighbor ordering."""
    points = np.asarray(points, dtype=float)
    n = len(points)

    if n < 2:
        return np.arange(n, dtype=int), pairwise_distances(points)

    if distance_matrix is None:
        D = pairwise_distances(points)
    else:
        D = np.asarray(distance_matrix, dtype=float)
        if D.shape != (n, n):
            raise ValueError("distance_matrix must have shape (n, n)")

    if n_starts is None or n_starts >= n:
        starts = range(n)
    else:
        starts = np.linspace(0, n - 1, n_starts, dtype=int)

    best_order = None
    best_len = np.inf

    for s in starts:
        visited = np.zeros(n, dtype=bool)
        order = [int(s)]
        visited[int(s)] = True

        for _ in range(n - 1):
            current = order[-1]
            candidates = np.where(~visited)[0]
            nxt = candidates[np.argmin(D[current, candidates])]
            order.append(int(nxt))
            visited[int(nxt)] = True

        order = np.asarray(order, dtype=int)
        curr_len = path_length(D, order, closed=closed)

        if curr_len < best_len:
            best_len = curr_len
            best_order = order

    return best_order, D  # type: ignore[return-value]


def two_opt_improve(
    D: np.ndarray,
    order: np.ndarray,
    closed: bool = True,
    max_passes: int = 20,
) -> np.ndarray:
    """Apply 2-opt local improvements to reduce path length."""
    order = np.asarray(order, dtype=int).copy()
    n = len(order)

    if n < 4:
        return order

    for _ in range(max_passes):
        improved = False

        for i in range(1, n - 1):
            j_end = n - 1 if closed else n - 2
            for j in range(i + 1, j_end + 1):
                a = order[i - 1]
                b = order[i]
                c = order[j]
                d = order[(j + 1) % n] if closed else order[j + 1]

                old_cost = D[a, b] + D[c, d]
                new_cost = D[a, c] + D[b, d]

                if new_cost + 1e-12 < old_cost:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True

        if not improved:
            break

    return order


def order_points_nn_2opt(
    t: np.ndarray,
    v: np.ndarray,
    n_starts: int | None = None,
    max_passes: int = 20,
    *,
    angular_t: bool = True,
    t_period: float = 2.0 * np.pi,
) -> dict:
    """
    Order cloud points by nearest-neighbor + 2-opt.

    Returns a dictionary with ordered coordinates and diagnostics.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    if t.shape != v.shape:
        raise ValueError("t and v must have the same shape")

    if angular_t:
        t_work = np.mod(t, t_period)
        D = _pairwise_theta_v_distances(t_work, v, t_period)
    else:
        t_work = t
        points = np.column_stack([t_work, v])
        D = pairwise_distances(points)

    points = np.column_stack([t_work, v])

    order_nn, D = nearest_neighbor_order(
        points,
        closed=True,
        n_starts=n_starts,
        distance_matrix=D,
    )
    order = two_opt_improve(D, order_nn, closed=True, max_passes=max_passes)

    ordered = points[order]
    t_ord = ordered[:, 0]
    v_ord = ordered[:, 1]

    if angular_t and len(t_ord) >= 2:
        t_ord_unwrapped = _unwrap_angles(t_ord, t_period)
        if (t_ord_unwrapped[-1] - t_ord_unwrapped[0]) < 0:
            t_ord = t_ord[::-1]
            v_ord = v_ord[::-1]
            order = order[::-1]
            order_nn = order_nn[::-1]

    return {
        "order": order,
        "order_nn": order_nn,
        "t_ord": t_ord,
        "v_ord": v_ord,
        "points": points,
        "distance_matrix": D,
        "length_nn": path_length(D, order_nn, closed=True),
        "length_refined": path_length(D, order, closed=True),
    }


def fit_closed_curve_from_cloud(
    t: np.ndarray,
    v: np.ndarray,
    *,
    n_starts: int | None = None,
    max_passes: int = 20,
    spline_degree: int = 3,
    smoothing: float = 1e-2,
    n_eval: int = 1000,
    angular_t: bool = True,
    t_period: float = 2.0 * np.pi,
):
    """
    Fit a closed spline from a polar cloud (v, t) via Cartesian coordinates.

    The returned callable takes parameter values u and returns a 2xN array
    containing [v(u), t(u)] where t is wrapped to [0, 2*pi).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.shape != v.shape:
        raise ValueError("t and v must have the same shape")

    # Ignore angular_t and t_period by design: ordering/fitting is done in
    # Cartesian coordinates to avoid angular seam issues.
    x = v * np.cos(t)
    y = v * np.sin(t)
    points_xy = np.column_stack([x, y])

    order_nn, D = nearest_neighbor_order(points_xy, closed=True, n_starts=n_starts)
    order = two_opt_improve(D, order_nn, closed=True, max_passes=max_passes)
    ordered_xy = points_xy[order]

    x_ord = ordered_xy[:, 0]
    y_ord = ordered_xy[:, 1]

    if not (np.isclose(x_ord[0], x_ord[-1]) and np.isclose(y_ord[0], y_ord[-1])):
        x_fit = np.r_[x_ord, x_ord[0]]
        y_fit = np.r_[y_ord, y_ord[0]]
    else:
        x_fit = x_ord
        y_fit = y_ord

    spl_xy, _ = make_splprep(
        [x_fit, y_fit],
        k=spline_degree,
        s=smoothing,
        bc_type="periodic",
    )

    def spline_polar(u_eval):
        u_arr = np.asarray(u_eval, dtype=float)
        x_eval, y_eval = spl_xy(u_arr)
        v_eval = np.sqrt(x_eval * x_eval + y_eval * y_eval)
        t_eval = np.mod(np.arctan2(y_eval, x_eval), 2.0 * np.pi)
        return np.vstack([v_eval, t_eval])

    return spline_polar
