import rebound
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import time
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


class FastSitnikovSimulation:
    """Fast, SciPy-based Sitnikov simulator with periodic focal-distance interpolation."""

    def __init__(
        self,
        e,
        n_focal_samples=4096,
        rtol=1e-7,
        atol=1e-10,
        max_step=np.inf,
        solver_method="RK45",
        auto_solver_trials=4,
        phi_time_window=20.0 * np.pi,
        focal_interp="linear",
    ):
        if not (0.0 <= e < 1.0):
            raise ValueError("e must satisfy 0 <= e < 1")
        if n_focal_samples < 16:
            raise ValueError("n_focal_samples must be at least 16")
        if focal_interp not in ("linear", "cubic"):
            raise ValueError("focal_interp must be 'linear' or 'cubic'")

        self.e = float(e)
        self.period = 2.0 * np.pi
        self.a = 0.5
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_step = float(max_step)
        self.phi_time_window = float(phi_time_window)
        self.focal_interp = focal_interp

        self.r_min = 0.5 * (1.0 - self.e)
        self.r_min_sq = self.r_min * self.r_min
        self.tau_min = np.pi * (self.r_min ** 1.5)

        self._n_focal_samples = int(n_focal_samples)
        self._t_grid = np.linspace(0.0, self.period, self._n_focal_samples, endpoint=False)
        self.focal_distance_array = self._build_focal_distance_array(self._t_grid)

        self._dt_grid = self.period / self._n_focal_samples
        self._inv_dt_grid = 1.0 / self._dt_grid
        self._r_periodic = np.concatenate((self.focal_distance_array, [self.focal_distance_array[0]]))
        self._t_periodic = np.linspace(0.0, self.period, self._n_focal_samples + 1)

        self._focal_spline = None
        if self.focal_interp == "cubic":
            self._focal_spline = CubicSpline(self._t_periodic, self._r_periodic, bc_type="periodic")

        self._escape_event = self._build_escape_event()

        if solver_method == "auto":
            self.solver_method = self._choose_fastest_solver(auto_solver_trials)
        else:
            self.solver_method = solver_method

    def _solve_kepler(self, M, tol=1e-14, max_iter=16):
        E = np.array(M, dtype=float, copy=True)
        for _ in range(max_iter):
            f = E - self.e * np.sin(E) - M
            fp = 1.0 - self.e * np.cos(E)
            dE = f / fp
            E -= dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    def _build_focal_distance_array(self, t_grid):
        M = np.mod(t_grid, self.period)
        E = self._solve_kepler(M)
        return self.a * (1.0 - self.e * np.cos(E))

    def _focal_distance_scalar(self, t):
        tau = t % self.period
        if self.focal_interp == "cubic":
            return float(self._focal_spline(tau))

        u = tau * self._inv_dt_grid
        i = int(u)
        frac = u - i
        r0 = self._r_periodic[i]
        r1 = self._r_periodic[i + 1]
        return r0 + frac * (r1 - r0)

    def focal_distance(self, t):
        tau = np.mod(t, self.period)
        if self.focal_interp == "cubic":
            return self._focal_spline(tau)
        return np.interp(tau, self._t_periodic, self._r_periodic)

    def _rhs(self, t, y):
        z, vz = y
        r = self._focal_distance_scalar(t)
        denom = (z * z + r * r) ** 1.5
        return (vz, -z / denom)

    def _escape_polynomial(self, z, vz):
        vz2 = vz * vz
        return 0.25 * vz2 * vz2 * (z * z + self.r_min_sq) - 1.0

    def _is_escaped(self, z, vz):
        return self._escape_polynomial(z, vz) >= 0.0

    def _build_escape_event(self):
        def event(t, y):
            return self._escape_polynomial(y[0], y[1])

        event.terminal = True
        event.direction = 1.0
        return event

    def _build_gated_crossing_event(self, t0):
        tau = self.tau_min
        t_left = t0 + 0.4 * tau
        t_right = t0 + 0.8 * tau
        inv_span = 1.0 / (t_right - t_left)

        def event(ti, yi):
            if ti < t_left:
                return 1.0
            if ti < t_right:
                x = (t_right - ti) * inv_span
                return x + (1.0 - x) * yi[0]
            return yi[0]

        event.terminal = True
        event.direction = -1.0
        return event

    def trajectory_fast(self, z0, vz0, t0, T, dt):
        if T <= 0.0:
            raise ValueError("T must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_start = float(t0)
        t_end = t_start + float(T)
        t_eval = np.arange(t_start, t_end, float(dt))
        if t_eval.size == 0 or t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        sol = solve_ivp(
            self._rhs,
            (t_start, t_end),
            (float(z0), float(vz0)),
            method=self.solver_method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
        )
        if not sol.success:
            raise RuntimeError(f"trajectory_fast integration failed: {sol.message}")

        return sol.y[0], sol.y[1]

    def stroboscopic(self, z, v):
        z0 = float(z)
        v0 = float(v)

        sol = solve_ivp(
            self._rhs,
            (0.0, self.period),
            (z0, v0),
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
        )
        if not sol.success:
            raise RuntimeError(f"stroboscopic integration failed: {sol.message}")

        return float(sol.y[0, -1]), float(sol.y[1, -1])

    def _phi_fast_impl(self, v, t, method, t_max, return_mod_period=True):
        # Note the precision for values of v under 1e-7 is really bad.
        # The integration breaks down.
        t0 = float(t)
        v0 = float(v)

        if v0 < 0.0:
            raise ValueError(f"Velocity must be non-negative, got v = {v}")
        if v0 <= 1e-7:
            # We increase the velocity to have a well-defined time integration.
            warnings.warn(f"Velocity {v0} is too small for reliable integration.")
            return v0, self._phi_fast_impl(1e-6, t, method, t_max, return_mod_period)[1]

        if self._is_escaped(0.0, v0):
            return None, None

        crossing_event = self._build_gated_crossing_event(t0)
        sol = solve_ivp(
            self._rhs,
            (t0, t0 + float(t_max)),
            (0.0, v0),
            method=method,
            events=(crossing_event, self._escape_event),
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
        )
        if not sol.success:
            raise RuntimeError(f"phi_fast integration failed: {sol.message}")

        t_cross = np.inf
        t_escape = np.inf

        if len(sol.t_events[0]) > 0:
            t_cross = float(sol.t_events[0][0])
            v_cross = float(sol.y_events[0][0][1])
        if len(sol.t_events[1]) > 0:
            t_escape = float(sol.t_events[1][0])

        if not np.isfinite(t_cross) or t_escape <= t_cross:
            return None, None

        if v_cross > 1e-8:
            raise RuntimeError("Detected crossing is not downward as expected")

        t_out = float(np.mod(t_cross, self.period)) if return_mod_period else t_cross
        return max(0.0, -v_cross), t_out

    def _choose_fastest_solver(self, trials):
        '''
        Not used by default because RK45 is generally faster for this problem.
        Run timing trials to choose the fastest ODE solver for phi_fast.
        '''
        candidates = ("DOP853", "RK45")
        trial_count = max(1, int(trials))
        v_samples = np.linspace(0.2, 1.0, trial_count)
        t_samples = np.linspace(0.0, self.period, trial_count, endpoint=False)
        timings = {}

        for method in candidates:
            start = time.perf_counter()
            for v0, t0 in zip(v_samples, t_samples):
                self._phi_fast_impl(v0, t0, method=method, t_max=2.0 * self.period)
            timings[method] = time.perf_counter() - start

        return min(timings, key=timings.get)

    def phi_fast(self, v, t, t_max=None, return_mod_period=True):
        if t_max is None:
            t_max = self.phi_time_window
        return self._phi_fast_impl(
            v=v,
            t=t,
            method=self.solver_method,
            t_max=t_max,
            return_mod_period=return_mod_period,
        )


    def phi_inv_fast(self, v, t, t_max=None, return_mod_period=True):
        if t_max is None:
            t_max = self.phi_time_window

        v_out, t_out = self._phi_fast_impl(
            v=v,
            t=-float(t),
            method=self.solver_method,
            t_max=t_max,
            return_mod_period=return_mod_period,
        )
        if v_out is None or t_out is None:
            return None, None
        
        return float(v_out), -float(t_out) % self.period if return_mod_period else -float(t_out)

    def fast_crossings_iterated(self, v, t, max_crossings=1000, t_max=None):
        """
        Count returns to z=0 by iterating phi_fast.

        Each iteration advances one return map step and therefore counts one
        crossing. If the global time budget is exhausted before escape and
        before reaching max_crossings, the method returns max_crossings.
        """
        if max_crossings <= 0:
            return 0

        v_curr = float(v)
        if v_curr < 0.0:
            raise ValueError(f"Velocity must be non-negative, got v = {v}")

        max_crossings_i = int(max_crossings)
        if t_max is None:
            t_max = max_crossings_i * self.phi_time_window
        t_budget = float(t_max)
        if t_budget <= 0.0:
            return 0

        t_curr = float(t)
        if self._is_escaped(0.0, v_curr):
            return 0

        t_end_total = t_curr + t_budget
        count = 0

        while count < max_crossings_i:
            remaining = t_end_total - t_curr
            if remaining <= 0.0:
                warnings.warn("fast_crossings reached time budget before max_crossings")
                return max_crossings_i

            v_next, t_next = self._phi_fast_impl(
                v=v_curr,
                t=t_curr,
                method=self.solver_method,
                t_max=remaining,
                return_mod_period=False,
            )

            if v_next is None or t_next is None:
                return count

            t_next_f = float(t_next)
            if t_next_f <= t_curr:
                raise RuntimeError("fast_crossings got a non-increasing return time")

            v_curr = float(v_next)
            t_curr = t_next_f
            count += 1

        return count

    def fast_crossings_chunked(self, v, t, max_crossings=1000, t_max=None):
        """
        Count z=0 crossings by chunked trajectory integration.

        Integration is performed over consecutive windows of length tau_min,
        counting a crossing whenever the endpoint values change sign or when
        an endpoint lands exactly on z = 0. Escape detection is checked during
        integration and stops the count early.

        Parameters
        ----------
        v : float
            Initial vertical velocity at z=0.
        t : float
            Initial time.
        max_crossings : int
            Maximum number of crossings to count before stopping. If the time
            budget is exhausted before the trajectory escapes or reaches this
            limit, the method returns max_crossings.
        t_max : float or None
            Total integration time budget across the whole trajectory.
            If None, defaults to max_crossings * self.phi_time_window.
        """
        if max_crossings <= 0:
            return 0

        v_curr = float(v)
        if v_curr < 0.0:
            raise ValueError(f"Velocity must be non-negative, got v = {v}")

        max_crossings_i = int(max_crossings)
        if t_max is None:
            t_max = max_crossings_i * self.phi_time_window
        t_budget = float(t_max)
        if t_budget <= 0.0:
            return 0

        t_curr = float(t)
        z_curr = 0.0
        count = 0

        if self._is_escaped(z_curr, v_curr):
            return 0

        t_end_total = t_curr + t_budget

        while t_curr < t_end_total and count < max_crossings_i:
            dt = min(self.tau_min, t_end_total - t_curr)
            if dt <= 0.0:
                break

            sol = solve_ivp(
                self._rhs,
                (t_curr, t_curr + dt),
                (z_curr, v_curr),
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                max_step=min(self.max_step, dt),
            )
            if not sol.success:
                raise RuntimeError(f"crossings_fast integration failed: {sol.message}")

            z_next = float(sol.y[0, -1])

            # z_next == 0.0 considered for robustness
            # If z_next == 0, there will no crossing in the next window
            if z_curr*z_next < 0 or z_next == 0.0:
                count = count + 1

            z_curr = z_next
            v_curr = float(sol.y[1, -1])
            t_curr = float(sol.t[-1])

            if self._is_escaped(z_curr, v_curr):
                break
        
        if t_curr >= t_end_total and count < max_crossings_i:
            warnings.warn("crossings_fast reached time budget before max_crossings")
            return max_crossings_i
        return count