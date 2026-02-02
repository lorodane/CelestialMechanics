import numpy as np
import matplotlib.pyplot as plt
import rebound
from src.integrator.integrate import initialize_simulation


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


