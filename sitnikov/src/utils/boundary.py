# Import my custom code from the editable package in sitnikov/src
from src.integrator.integrate import *

#Import other libraries
import numpy as np
import matplotlib.pyplot as plt
import rebound
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


'''
Users should only need to use B1_v_func, B2_v_func.
And B2_return_time_lower_bound for diagnosing B2 boundary.
'''

def B2_v_func(e, dv = 1e-3, N_t = 100):
    '''
    Returns a function v(t) corresponding to boundary B2 (certain return).
    The radial distance from the true boundary will be less than dv unless 
    e is high (0.70 for default params).

    N_t controls the number of sampled points and dv controls the precision
    of each sampled point.
    '''

    if e >= 0.70:
        raise Warning("High value of e. Consider increasing dv, N_t for high precision")
    
    return create_cubicspline(B2_tv_array(dv, N_t, e))

def B1_v_func(e, dv = 1e-3, N_t = 100):
    '''
    Returns a function v(t) corresponding to boundary B1 (certain escape).
    The radial distance from the true boundary will be less than dv unless 
    e is high (0.70 for default params).

    N_t controls the number of sampled points and dv controls the precision
    of each sampled point.
    '''

    if e >= 0.70:
        raise Warning("High value of e. Consider increasing dv, N_t for high precision")
    
    return create_cubicspline(B1_tv_array(dv, N_t, e))
    

def _integrate_back_to_crossing(e, z0, v0, t0):
        '''
        Returns the first crossing velocity, time.
        Note the time will probably be negative.
        '''
        # We initialize with negative velocity at negative time to integrate backwards
        sim = initialize_simulation(e = e, z = z0, z_dot = -v0, t = -t0)
        
        # Now we integrate until we have intersection with z = 0
        dt = 0.1 # Initial timestep where we look for intersection with z = 0

        sim_prev = sim.copy()
        sim.integrate(sim.t + dt)

        while sim.particles[2].z > 0:
            sim_prev = sim.copy()
            sim.integrate(sim.t + dt)

        # Now we have that the crossing occurs between sim_prev.t and sim.t,
        # so we refine with brentq    

        def z_func(t):
            sim_loc = sim_prev.copy()
            sim_loc.integrate(t, exact_finish_time=1)
            return sim_loc.particles[2].z
        
        t_cross = brentq(z_func, sim_prev.t, sim_prev.t + dt, xtol = 1e-14)

        sim_prev.integrate(t_cross)
        v_final = sim_prev.particles[2].vz

        # Change signs again
        return (-v_final, -t_cross)



def B2_tv_array(dv, N_t, e):
    z0 = (np.sqrt(2)/4 * e / dv)**(2/5)
    r_2 = 1/2* (1 + e)
    v2 = np.sqrt(2/np.sqrt(z0**2 + r_2**2))

    # We store the points in a list
    # First the inner boundary (B2) inside of which we have certain return
    B2_tv_arr = np.zeros((2, N_t))

    for (i,t) in zip(range(N_t), np.linspace(0, 2*np.pi, N_t, endpoint=False)):

        v_cross, t_cross = _integrate_back_to_crossing(e, z0, v2, t)

        B2_tv_arr[1][i] = v_cross
        B2_tv_arr[0][i] = t_cross % (2*np.pi)

    print("Points on B2 have a return time of at least {0:.3e}".format(B2_return_time_lower_bound(dv, N_t, e)))
    
    return B2_tv_arr

def B1_tv_array(dv, N_t, e):
    z0 = (np.sqrt(2)/4 * e / dv)**(2/5)
    r_1 = 1/2* (1 - e)
    v1 = np.sqrt(2/np.sqrt(z0**2 + r_1**2))

    # We store the points in a list
    # First the inner boundary (B2) inside of which we have certain return
    B1_tv_arr = np.zeros((2, N_t))

    for (i,t) in zip(range(N_t), np.linspace(0, 2*np.pi, N_t, endpoint=False)):

        v_cross, t_cross = _integrate_back_to_crossing(e, z0, v1, t)

        B1_tv_arr[1][i] = v_cross
        B1_tv_arr[0][i] = t_cross % (2*np.pi)
    
    return B1_tv_arr

def create_cubicspline(tv_arr):
    '''
    Computes the interpolated value v(t) 
    given an array of samples tv_arr.
    tv_arr[0] is the array of t values (increasing with one wrap-around jump).
    tv_arr[1] is the array of v values.
    The function is 2pi periodic.
    '''
    t_samples = tv_arr[0]
    v_samples = tv_arr[1]
    
    # Handle wrap-around: find the jump where t decreases
    # Find the index where t[i+1] < t[i]
    diffs = np.diff(t_samples)
    jump_indices = np.where(diffs < 0)[0]
    
    if len(jump_indices) > 0:
        # We take the first jump found
        idx = jump_indices[0] + 1
        # Reorder arrays to be strictly increasing in t
        t_sorted = np.concatenate([t_samples[idx:], t_samples[:idx]])
        v_sorted = np.concatenate([v_samples[idx:], v_samples[:idx]])
    else:
        # Already sorted or all values are same (unlikely)
        t_sorted = t_samples
        v_sorted = v_samples
        
    # Close the periodic loop to ensure CubicSpline(bc_type = 'periodic') works
    t_closed = np.concatenate([t_sorted, np.array([t_sorted[0] + 2*np.pi])])
    v_closed = np.concatenate([v_sorted, np.array([v_sorted[0]])])
    
    # Create the periodic cubic spline
    cs = CubicSpline(t_closed, v_closed, bc_type='periodic')
    
    return cs


def _return_time_lower_bound(e, z0, v0):
    '''
    Quick lower bound for return time.
    Assumes:
    z0, v0 > 0
    Return is guaranteed because 
    v <= v_esc(primaries at perihelion)  
    '''

    if z0 <= 0 or v0 <= 0:
        raise ValueError(f"Invalid input: expected z0 > 0 and v0 > 0, got z0={z0}, v0={v0}")
    
    a = 1/2 * (1-e)
    r0 = np.sqrt(z0**2 + a**2)

    if 1/2 * v0**2 - 1/r0 >= 0:
        raise ValueError(f"Invalid input: Return not guaranteed when setting primaries at perihelion, got z0 = {z0}, v0 = {v0}")
    
    rM = 1/ (1/r0 - 1/2 * v0**2 )
    u0 = r0/rM
    # Time to slow-down lower bound
    T = 1/np.sqrt(2) * rM**(3/2) * (np.pi/2 - np.arcsin(np.sqrt(u0)) + np.sqrt(u0)*np.sqrt(1-u0))

    return 2*T # going up and falling back down


def B2_return_time_lower_bound(dv, N_t, e):
    z0 = (np.sqrt(2)/4 * e / dv)**(2/5)
    r_2 = 1/2* (1 + e)
    v2 = np.sqrt(2/np.sqrt(z0**2 + r_2**2))

    return _return_time_lower_bound(e, z0, v2)

    

