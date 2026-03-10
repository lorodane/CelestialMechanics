# Experiment Template

## Description
Here we want to make some global plots of escape time. Note than none of the features will be seen with their utmost detail in these plots, so they should only serve as a general guide to macro features of escape time. 

In particular, due to computational constraints, orbits which take a long time to return are directly classified as "escaping". This means that the behaviour at the border is not captured with nuance. On the other hand, the resolution is relatively low so it does not allow one to see the details of escape time close to the central island of stability. And additionally, any small islands of stability are likely to be missed by these plots because of their low resolution 


## Execution and Parameters
We choose the following list of eccentricities: [0.05, 0.2, 0.6, 0.9].

We make plots with boundaries that are computed with parameters dv = 1e-3, N_t = 100.
The plots have irregular grids in the region enclosed by the boundary with N_t = 40, N_v = 20, max_ret = 20, t_max = 100.


## Results Summary
- Key findings or observations.

