
survivors_final plot were generated with e = 0.5.
with the sprinkler method, sampling points randomly from a segment in t=0, between the outer edge of the main central island and the escape boundary.


For the context of the chaotic_saddle_sprinkler2.ipynb file, note with default precision the final points were all very close to the main island (suspiciously close) and a full 26.8% of randomly sampled points survived for up to N_STEPS = 100, that seems excessive and possibly due to systematic integration error.

Let me test that again with r_tol = 1e-9, a_tol = 1e-12

Note that the files sprinkler_*_high_precision1.png correspond to N0 = 500, N_STEPS = 200, rtol = 1e-10, atol = 1e-13, E = 0.5
For those circumstances we surprisingly get a survival rate of 27%, which is ridiculous.

I get it now, the issue is that those points are actually inside of the main stable island.