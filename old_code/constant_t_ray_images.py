# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:58:24 2025

@author: leona
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import sitnikov.tools.sitnikov_integrator as sit
import sitnikov.tools.geometrical_helpers as geo

sit.set_eps(0.5)

n_t = 12
n_v = 50
v_max = 1.3

t = np.linspace(0, 2*np.pi, n_t)
v = np.linspace(0, v_max, n_v)

# remember the order is always (t,v)
output = np.zeros((n_t,2,n_v))
for i in range(n_t):
    for j in range(n_v):
        output[i][0][j], output[i][1][j] = sit.phi(t[i],v[j])

fig = None
for i in range(n_t-1):
    fig = sit.polar_plot(output[i][0], output[i][1], 
                         fig=fig, title = "Images of constant t rays for e={0}".format(sit.eps()))


print(output[:,:,0])

ax = fig.get_axes()[0]
legend_handles = ["t = {0:.2f}".format(t_val) for t_val in t]
ax.legend(legend_handles)





