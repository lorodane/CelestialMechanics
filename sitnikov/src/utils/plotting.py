from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.integrator.integrate import FastSitnikovSimulation
from src.integrator.integrate import initialize_simulation



def save_figure(fig: Figure, name: str, folder_path, dpi: int = 300, bbox_inches: str = "tight", extension: str = "png"):
    """
    Save a Matplotlib figure to a folder and return the saved path.

    The helper creates the target folder if needed and raises FileExistsError
    if the final file already exists.
    """

    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    extension = extension.lstrip(".")
    save_path = folder / f"{name}.{extension}"

    if save_path.exists():
        raise FileExistsError(f"Figure already exists at {save_path}")

    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    return save_path


# Function to diagnose orbits
def plot_position_sitnikov(sim=None, T=None, dt=None, e = None, t=None, z=None, z_dot=None, v=None):
    '''
    Plots a 1D plot of position over time given the simulation variable
    time range from 0 to dt*np.floor(T/dt).
    Can be called with multiple keyword argument combinations.
    Modes:
    1. plot_position_sitnikov(sim=s, T=100, dt=0.1)
    2. plot_position_sitnikov(e = 0.5, t=0, z=1, z_dot=0, T=100, dt=0.1)
    3. plot_position_sitnikov(e = 0.5, v=1.5, t=0, T=100, dt=0.1)
    '''

    # 1. Global Requirements
    if T is None or dt is None:
        raise ValueError("Arguments 'T' and 'dt' are required.")

    # 2. Define Valid Modes (Check which groups of args are fully present)
    # Mode 1: Existing Simulation
    has_sim = sim is not None and all(x is None for x in [e, t, z, z_dot, v])
    
    # Mode 2: Velocity Initialization (requires e, v, t)
    has_v_init = all(x is not None for x in [e, v, t]) and all(x is None for x in [sim, z, z_dot])
    
    # Mode 3: State Initialization (requires e, z, z_dot, t)
    has_z_init = all(x is not None for x in [e, z, z_dot, t]) and all(x is None for x in [sim, v])
    
    if not has_sim and not has_v_init and not has_z_init:
        raise ValueError("Invalid arguments. Please provide only (sim, T, dt) OR (e, v, t, T, dt) OR (e, z, z_dot, t, T, dt).")

    if initialize_simulation is None:
        raise ImportError("plot_position_sitnikov requires rebound-backed simulation support, which is unavailable in this environment.")

    if has_sim:
        sim2 = sim.copy() # type: ignore
    elif has_v_init:
        sim2 = initialize_simulation(e = e, z = 0, z_dot = v, t = t)
    elif has_z_init:
        sim2 = initialize_simulation(e = e, z = z, z_dot = z_dot, t = t)

    N = int(np.floor(T/dt)) + 1 # for each multiple dt*i \leq T
    z_arr = np.zeros((N,))
    for i in range(N):
        z_arr[i] = sim2.particles[2].z # type: ignore
        sim2.integrate(sim2.t + dt) # type: ignore
    
    plt.plot(np.arange(N)*dt, z_arr)
    plt.show()


def polar_scatter(theta, r):
    theta = np.asarray(theta)
    r = np.asarray(r)

    if theta.shape != r.shape:
        raise ValueError("'theta' and 'r' must have the same shape.")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.scatter(theta, r)

    return fig, ax

def phase_space_plot(e, N_v, N_t, N_it, random_dist=False, spacing=0, area_preserving_dist=False):
    '''
    Plots a phase space plot. Uses FastSitnikovSimulation to compute the trajectories.
    It plots the phase space of the Sitnikov problem for a given eccentricity e, number of initial velocities N_v, number of initial times N_t, and number of iterations N_it. The plot is in polar coordinates, where the angle corresponds to time modulo the period and the radius corresponds to velocity.
    If random_dist is True, the initial conditions are sampled uniformly at random. Otherwise, they are sampled on a regular grid.
    If spacing is greater than 0, it enforces a minimum spacing between plotted points in
    the (t mod period, v) space. This can help reduce overplotting and improve performance, but it also makes the code significantly slower.
    If area_preserving_dist is True, the initial conditions are sampled in a way that ensures a uniform distribution over area in the (t mod period, v) space. This means that v^2 is sampled uniformly instead of v. This can help reduce overplotting near the origin and provide a more accurate representation of the phase space, but it also makes the code significantly slower.
    '''
    # Note spacing makes the code significantly slower.
    if N_v <= 0 or N_t <= 0 or N_it <= 0:
        raise ValueError("N_v, N_t, and N_it must be positive")
    if spacing < 0:
        raise ValueError("spacing must be non-negative")

    sim = FastSitnikovSimulation(e=e)
    period = 2.0 * np.pi
    v_max = 2.0 / np.sqrt(1.0 - e) # This is a very rough estimate. It would be nice to improve it in the future.


    if area_preserving_dist:
        # We ensure uniform distribution over area by sampling v^2 and t uniformly, and then taking the square root of the v^2 samples.
        if random_dist:
            rng = np.random.default_rng()
            v_samples = np.sqrt(np.sort(rng.uniform(0.0, v_max**2, size=int(N_v))))
            t_samples = np.sort(rng.uniform(0.0, period, size=int(N_t)))
        else:
            v_samples = np.sqrt(np.linspace(0.0, v_max**2, int(N_v) + 2)[1:-1])
            t_samples = np.linspace(0.0, period, int(N_t), endpoint=False)
    else:
        if random_dist:
            rng = np.random.default_rng()
            v_samples = np.sort(rng.uniform(0.0, v_max, size=int(N_v)))
            t_samples = np.sort(rng.uniform(0.0, period, size=int(N_t)))
        else:
            v_samples = np.linspace(0.0, v_max, int(N_v) + 2)[1:-1]
            t_samples = np.linspace(0.0, period, int(N_t), endpoint=False)



    plotted_points = np.empty((0, 2), dtype=float)
    kept_points_t = []
    kept_points_v = []

    for t0 in t_samples:
        for v0 in v_samples:
            trajectory = [(float(v0), float(t0))]
            v_curr = float(v0)
            t_curr = float(t0)

            for _ in range(int(N_it)):
                v_next, t_next = sim.phi_fast(v=v_curr, t=t_curr, return_mod_period=True)
                if v_next is None or t_next is None:
                    break
                trajectory.append((float(v_next), float(t_next)))
                v_curr = float(v_next)
                t_curr = float(t_next)

            if len(trajectory) < int(N_it + 1):
                continue

            trajectory_arr = np.asarray(trajectory, dtype=float)
            if spacing > 0 and plotted_points.size > 0:
                delta_t = np.abs(trajectory_arr[:, None, 1] - plotted_points[None, :, 1])
                delta_t = np.minimum(delta_t, period - delta_t)
                delta_v = trajectory_arr[:, None, 0] - plotted_points[None, :, 0]
                distances = np.sqrt(delta_v * delta_v + delta_t * delta_t)
                if np.any(np.min(distances, axis=1) < spacing):
                    continue

            kept_points_v.extend(trajectory_arr[:, 0])
            kept_points_t.extend(trajectory_arr[:, 1])
            plotted_points = np.vstack((plotted_points, trajectory_arr))

    # Plotting logic here
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
    if kept_points_t:
        ax.scatter(kept_points_t, kept_points_v, s=1, color="black")
    ax.set_rlim(0.0, v_max)
    return fig, ax

