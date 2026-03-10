# Boundary Algorithm

## Description
We implement an algorithm to efficiently compute precise approximations for the boundary D0 which separates points that will return to z=0 in future time from those that will not return to z=0.

## Results Summary
The algorithm was implemented, tested and a lower bound for the return time was also implemented and tested to go along with the algorithm. Both are now stored in src/utils/boundary.py for future use.

The computed boundaries (B2, B1) for a range of eccentricities, with default parameters dv = 1e-3, N_t = 100, have been plotted and stored in ./plots.

