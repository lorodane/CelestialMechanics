# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:16:39 2025

@author: leona
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import sitnikov.tools.sitnikov_integrator as sit
import sitnikov.tools.geometrical_helpers as geo

sit.set_eps(0.5)

v_min = 0.1
v_max = 1.3
n_v = 13
dv = (v_max-v_min)/(n_v-1)

t = np.linspace(0, 2*np.pi, 100)
output = np.zeros((n_v,2,t.size))

# remember the order is always (t,v)
for i in range(n_v):
    for j in range(t.size):
        output[i][0][j], output[i][1][j] = sit.phi(t[j],v_min + i*dv)

fig = None
for i in range(n_v):
    fig = sit.polar_plot(output[i][0], output[i][1], 
                         fig=fig, title = "Images of constant v curves for e ={0}\nPoints are phi(0,v)".format(sit.eps()))

for i in range(n_v):
    fig.get_axes()[0].plot(output[i][0][0],output[i][1][0],'o') 

print(output[:,:,0])

ax = fig.get_axes()[0]
legend_handles = ["v = {0:.2f}".format(v_min + i*dv) for i in range(n_v)]
ax.legend(legend_handles)










