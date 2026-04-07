import numpy as np
from scipy.interpolate import make_splprep


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
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-start greedy nearest-neighbor ordering."""
    points = np.asarray(points, dtype=float)
    n = len(points)

    if n < 2:
        return np.arange(n, dtype=int), pairwise_distances(points)

    D = pairwise_distances(points)

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
) -> dict:
    """
    Order cloud points by nearest-neighbor + 2-opt.

    Returns a dictionary with ordered coordinates and diagnostics.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    if t.shape != v.shape:
        raise ValueError("t and v must have the same shape")

    points = np.column_stack([t, v])

    order_nn, D = nearest_neighbor_order(points, closed=True, n_starts=n_starts)
    order = two_opt_improve(D, order_nn, closed=True, max_passes=max_passes)

    ordered = points[order]
    t_ord = ordered[:, 0]
    v_ord = ordered[:, 1]

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
) -> dict:
    """
    Order an unordered 2D cloud and fit a closed parametric spline.

    Returns ordered points, smooth curve samples, and diagnostics.
    """
    ordered_info = order_points_nn_2opt(t, v, n_starts=n_starts, max_passes=max_passes)
    t_ord = ordered_info["t_ord"]
    v_ord = ordered_info["v_ord"]

    # make_splprep with periodic boundary conditions requires identical
    # first/last samples. Close the sequence explicitly if needed.
    if not (np.isclose(t_ord[0], t_ord[-1]) and np.isclose(v_ord[0], v_ord[-1])):
        t_fit = np.r_[t_ord, t_ord[0]]
        v_fit = np.r_[v_ord, v_ord[0]]
    else:
        t_fit = t_ord
        v_fit = v_ord

    spl, u = make_splprep(
        [t_fit, v_fit],
        k=spline_degree,
        s=smoothing,
        bc_type="periodic",
    )

    u_fine = np.linspace(u[0], u[-1], n_eval)
    t_smooth, v_smooth = spl(u_fine)

    ordered_info.update(
        {
            "spline": spl,
            "u": u,
            "u_fine": u_fine,
            "t_smooth": t_smooth,
            "v_smooth": v_smooth,
        }
    )
    return ordered_info
