import rebound
import numpy as np
from scipy.optimize import brentq
import warnings

def initialize_simulation(e, z, z_dot, t):
    '''
    Returns rebound simulation for Sitnikov problem given initial conditions.
    sim.integrator = ias15 by default, so it can handle any timescale
    and deal with high-precision long-term integration
    '''
    # Compute t mod 2pi
    k = np.floor(t/(2*np.pi))
    t2 = t - k*2*np.pi

    a = 1/2
    m = 1/2
    x_ph = a*(1-e)
    v_ph = 1/2*np.sqrt((1+e)/(1-e))

    # Prepare simulation and set primaries
    # to initial position at t/t2
    sim = rebound.Simulation()
    sim.add(m = 1/2, x = -x_ph, y = 0, z = 0,
                    vx = 0, vy = -v_ph, vz = 0)
    sim.add(m = 1/2, x = x_ph, y = 0, z = 0,
                    vx = 0, vy = v_ph, vz = 0)
    sim.integrate(t2)

    # we set time to t manuallly
    sim.t = t

    sim.add(m = 0, x = 0, y = 0, z = z,
                   vx = 0, vy = 0, vz = z_dot)
    
    return sim

def phi(e, v, t, t_max = 100):
    '''
    Pair of return (velocity, time) when it exists.
    Note v has to be a non-negative velocity.

    When the points don't return in less than max_time = 1000,
    we simply return (None, None)
    '''

    if v == 0:
        return (0,t) # Note with this definition the function has a discontinuity at zero

    if v<0:
        raise ValueError(f"Velocity must be non-negative, got v = {v}")

    sim = initialize_simulation(e=e, z=0, z_dot=v, t=t)
    
    # Coarse search
    r_min = 0.5*(1-e)
    # We ensure dt < T_ret/2, to avoid skipping two roots z = 0
    # Experimentally, this is confirmed
    dt = min(0.1, 2*v*r_min**2) 
    t_start = t
    
    while sim.t < t_start + t_max:
        sim_prev = sim.copy()

        sim.integrate(sim.t + dt)        
        
        if sim.particles[2].z < 0: 
            
            # The crossing time must be between t1 and t1 + dt 
            t1 = sim_prev.t

            # Refine with brentq
            def z_at_time(tau):
                s = sim_prev.copy()
                s.integrate(tau)
                return s.particles[2].z # type: ignore
            
            t_cross = brentq(z_at_time, t1, t1 + dt, xtol = 1e-18, rtol = 1e-15)  # type: ignore
            sim_final = sim_prev.copy()
            sim_final.integrate(t_cross)
            v_cross = sim_final.particles[2].vz # type: ignore
            
            return (-v_cross, t_cross)
    
    return (None, None)

def number_returns_before_escape(e, v0, t0, max_ret = 100, t_max = 100):
    '''
    Returns the number of returns to z=0 (capped at max_ret)
    before the particle takes t_max or longer to return.
    '''
    i = 0
    v = v0
    t = t0

    while v is not None and i < max_ret:
        v, t = phi(e, v, t, t_max = t_max)
        i += 1
    
    if v is None:
        return i - 1
    else:
        return max_ret

def phi_inv(e, v, t, t_max = 100):
    v1, t1 = phi(e, v, -t, t_max = t_max)
    if t1 == None: # If one is None, both are
        return (None, None)
    return v1, -t1

def check_escape(e, v, t, n, t_max=100):
    """
    Returns True if the particle escapes after n iterations.
    Returns False otherwise.
    """
    current_v = v
    current_t = t
    
    for _ in range(n):
        res = phi(e, current_v, current_t, t_max=t_max)
        if res[0] is None:
            return True
        current_v, current_t = res
        
    return False