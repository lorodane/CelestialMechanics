import tools.sitnikov_integrator as sit
from tools.sitnikov_integrator import phase_velocity
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

T = 6000
e = 0.5
sit.set_eps(e)

t0 = 0
v0 = 2.552
z0 = np.array((0, v0))

ivp_sol = solve_ivp(
    fun = phase_velocity,
    t_span = (t0, t0 + T),
    y0 = z0,
    dense_output = True
)

plt.plot(ivp_sol.t, ivp_sol.y[0])
plt.title(f"Trajectory for t0 = {t0}, v0 = {v0}")
plt.xlabel("t")
plt.ylabel("z")


import re
from pathlib import Path

def next_figure_number(fig_dir="figures", prefix="figure#"):
    p = Path(fig_dir)
    p.mkdir(parents=True, exist_ok=True)
    used = set()
    pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)(?:\..*)?$')
    for f in p.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                used.add(int(m.group(1)))
    n = 0
    while n in used:
        n += 1
    return n

n = next_figure_number()
plt.savefig(f"figures/figure#{n}")
plt.show()


