
import time
import tools.sitnikov_integrator as sit
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
from pathlib import Path

start = time.time()

e = 0.5
sit.set_eps(e)


t0 = 0
v_arr = np.linspace(2.5, 3, 200)

t_esc_arr = np.zeros(np.size(v_arr))

Tmax = 600

for i in range(v_arr.size):
    t_esc_arr[i] = sit.escape_time(t0, v_arr[i], Tmax)


print(v_arr)
print(t_esc_arr)
plt.plot(v_arr, t_esc_arr)

end = time.time()

print("Time to execute: ", end-start)

plt.show()


# # mask invalid/missing values (optional)
# data = np.ma.masked_invalid(t_esc_arr)

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))

# # Plot as colored points (each sample -> one marker)
# # Flatten grids / data and discard invalid samples so only actual evaluated points are plotted
# T, R = np.meshgrid(t_arr, v_arr, indexing='ij')   # shapes (nt, nv)
# T_flat = T.ravel()
# R_flat = R.ravel()
# data_flat = t_esc_arr.ravel()
# valid = np.isfinite(data_flat)
# T_plot = T_flat[valid]
# R_plot = R_flat[valid]
# data_plot = data_flat[valid]

# # choose a normalization (optional) so colors map consistently
# norm = Normalize(vmin=np.nanmin(data_plot), vmax=np.nanmax(data_plot))

# # scatter: c= maps scalar -> colors; s controls marker size
# sc = ax.scatter(T_plot, R_plot, c=data_plot, cmap='viridis', norm=norm, s=20, marker='o', edgecolors='face')

# # adjust radial limits
# ax.set_ylim(0, v_arr.max())

# # colorbar with label (pass the PathCollection returned by scatter)
# cbar = fig.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('Escape time')

# ax.set_title("Escape time (polar)")







# # plt.plot(v_arr, t_esc_arr)
# # plt.title("Escape time in terms of v for t0 = 0")
# # plt.xlabel("v")
# # plt.ylabel("$t_{esc}$")

# def next_figure_number(fig_dir="figures", prefix="figure#"):
#     p = Path(fig_dir)
#     p.mkdir(parents=True, exist_ok=True)
#     used = set()
#     pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)(?:\..*)?$')
#     for f in p.iterdir():
#         if f.is_file():
#             m = pattern.match(f.name)
#             if m:
#                 used.add(int(m.group(1)))
#     n = 0
#     while n in used:
#         n += 1
#     return n

# n = next_figure_number()
# plt.savefig(f"figures/figure#{n}")
# plt.show()


