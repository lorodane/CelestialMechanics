
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import sitnikov.tools.sitnikov_integrator as sit
import sitnikov.tools.geometrical_helpers as geo

start = time.time()

sit.set_eps(0.7)

tbv = np.load("boundary_test1/e0_5__v1_eps0_1__deltmax_100.npy")

fig = sit.polar_plot(tbv[0], tbv[1], title = "Close-up boundary")

delta = 0.2

def inner_perimeter(tbv, delta):
    n = tbv.shape[1]
    sbv = np.zeros((2,n))
    for i in range(n):
        if i != n-1:        
            p1 = geo.to_cartesian(tbv[:,i])
            print("Position of p1 is ", p1)
            p2 = geo.to_cartesian(tbv[:,i+1])
            dp = p2-p1
        else:
            p1 = geo.to_cartesian(tbv[:,n-1])
            p2 = geo.to_cartesian(tbv[:,n-2])
            dp = p1-p2
        
        ndp = np.sqrt(dp[0]**2 + dp[1]**2)
        rdp = np.array([-dp[1], dp[0]])/ndp
        
        print(rdp)
        q = p1 + delta*rdp
        print(q)
        
        sbv[:,i] = geo.to_polar(q)
        print(sbv[:,i])
    return sbv

def reflect(tbv):
    tbv[0,:] = -tbv[0,:]
    return tbv

sbv = inner_perimeter(tbv, delta)


sit.polar_plot(sbv[0], sbv[1], fig = fig)

tbv1 = reflect(tbv)
sbv1 = reflect(sbv)

sit.polar_plot(tbv1[0], tbv[1], fig = fig)
sit.polar_plot(sbv1[0], sbv[1], fig = fig)

ax = fig.get_axes()[0]




t0_line = geo.Line.from_point_slope((0,0), 0)

wait = input("Press Enter to continue.")




                











