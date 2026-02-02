# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:09:49 2025

@author: leona
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import tools.sitnikov_integrator as sit
import tools.geometrical_helpers as geo

e = 0.5
sit.set_eps(0.5)

D0 = np.load(file = "sitnikov/boundary_test1/e0_5__v2_eps0_001__deltmax_100.npy")
n = D0.shape[1]
D0 = np.concatenate((D0[:, n//2:], D0[:, 0:n//2+1]), axis = 1)
print(D0.shape)
fig = sit.polar_plot(D0[0], D0[1], title = "Boundary for e=0.5, dt_esc = 100")

def reflect(tbv):
    sbv = np.zeros(tbv.shape)
    sbv[0,:] = -tbv[0,:]
    sbv[1,:] = tbv[1,:]
    return sbv

def inner_perimeter(tbv, delta):
    n = tbv.shape[1]
    sbv = np.zeros((2,n))
    for i in range(n):
        if i != n-1:        
            p1 = geo.to_cartesian(tbv[:,i])
            p2 = geo.to_cartesian(tbv[:,i+1])
            dp = p2-p1
        else:
            p1 = geo.to_cartesian(tbv[:,n-1])
            p2 = geo.to_cartesian(tbv[:,n-2])
            dp = p1-p2
        
        ndp = np.sqrt(dp[0]**2 + dp[1]**2)
        rdp = np.array([-dp[1], dp[0]])/ndp
        
        q = p1 + delta*rdp
        
        sbv[:,i] = geo.to_polar(q)
    return sbv    

def plot_on_top(bv, fig):
    sit.polar_plot(bv[0], bv[1], fig=fig)


delta = 0.2

D1 = reflect(D0)
D0_in = inner_perimeter(D0, delta)
D1_in = reflect(D0_in)


plot_on_top(D1[:, 49:54], fig)
plot_on_top(D0_in[:, 45:51], fig)
plot_on_top(D1_in[:, 45:51], fig)


plt.legend(["D0", "D1", "D0_in", "D1_in"])


# compute image of sides of R

# compute the image of D0_in

# left = 47
# right = 49
# k = 20
# im = np.zeros((2, k*(right-left)))

# bv = D1_in

# for i in range(left, right):
#     p1 = geo.to_cartesian((bv[0][i], bv[1][i]))
#     p2 = geo.to_cartesian((bv[0][i+1], bv[1][i+1]))
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     for i2 in range(k):
#         j = (i-left)*k + i2
#         p = (k - i2)/k * p1 + i2/k * p2 
#         p = geo.to_polar(p)
#         im[0][j], im[1][j] = sit.phi(p[0], p[1])

# plot_on_top(im, fig)







# compute intersections between sides of R
# horizontal line
h = geo.Line.from_point_slope((0,0), 0)

# compute indices for each intersection
ex = [-1, -1]
extremes = {"D0": ex, "D1": ex, "D0_in": ex, "D1_in": ex}

# intersect D0 with h to get the intersection of D0 with D1

print(D0[:,0])
print(D0[:,-1])

inter = geo.intersection(h, D0)


print(inter)


extremes["D0"][0] = 0
extremes["D1"][1] = n-1

# Intersect D0_in with D1_in
I1 = geo.intersection(h, D0_in)

# We choose to orient D0_in in the same sense as D0
extremes["D0_in"][1]["i"] = I1[0]
extremes["D0_in"][1]["i"] = 0

D0_delta_i1 = I1[0]
D0_delta_r1 = I1[1]
D1_delta_i1 = D0_delta_i1
D1_delta_r1 = D0_delta_r1


D0_delta_i2 = -1
D0_delta_r2 = -1
for i in range(D0_in.shape[1]):
    Ddel = D0_in
    i_next = (i+1)%D0_in.shape[1]
    p1 = geo.to_cartesian((Ddel[0][i], Ddel[1][i]))
    p2 = geo.to_cartesian((Ddel[0][i_next], Ddel[1][i]))
    for j in range(D1.shape[1]):
        j_next = (j+1)%D1.shape[1]
        q1 = geo.to_cartesian((D1[0][j], D1[1][j]))
        q2 = geo.to_cartesian((D1[0][j_next], D1[1][j_next]))
        
        q = geo.segment_intersection(p1,p2,q1,q2)
        if q == -1:
            next
        if q[0] < 0:
            next
        
# intersect D0_delta with D1

n = D0_in.shape[1]
m = D1.shape[1]



# reflect



# compute image of the three sides of R

# First just the boundary points

# Image of D0_in
# Image of the other two sides

# Check visually for intersections with R



# Implement automatic detection of intersections, compute list
# of intersections for D0_in


















