I have tried to compute the chaotic saddle once and the results gave me points which were mostly around the central stable island.

This makes me suspect that most almost all points which are invariant are around stable islands.


phi_fast really allows me to make much more detailed escape time plots.

There a bunch of things in the back of my mind that I would want to do, but let's go from more related to my main goal to more unrelated.

Let's apply the sprinkler method again with higher N_STEPS (stability threshold)

It seems like long orbits are all over the place, sometimes moving what seems to be quite far from the stable central-island boundary. They move in ways that seem to have no pattern (irregular)

It totally seems like a majority of points that survive for a long time when we choose N_STEPS = 40 have orbits which come close to the central island.

For N_STEPS = 100 we have an even larger fraction of the stable orbits that "shadow" the boundary of the central island. Interestingly, none of them shadow the secondary island.

- compute the chaotic saddle with more points and higher crossings threshold (done, but I could do it with even more points)
- Look at where the iterates seem to concentrate for specific orbits (done)

A note is that precision might be quite critical if we try to determine the actual probabilities of landing in orbits that have a certain degree of long term stability.

Let's do the following:

- Look at the sprinkler method applied to points which are close to the secondary island

On a first computation, applying the sprinkler method to points on t=pi between the main and secondary island, we see the trajectories that survive for N_STEPS = 50 are still those that stick to the main island, instead of the secondary island.

I am running this again with N_STEPS = 100 to see if any outliers appear that shadow the secondary island.


So right now we have experimental evidence that:
+ Points that are stable for a long time mostly (in a measure sense) shadow the outer irc of the main island
+ Separate unstable fixed points or the secondary island don't seem to contribute significantly to the population of points with long-term stability


I am getting suspicious of the results, let me test if the precision of the integration method affects the results.

I increased the precision and I am still getting really high survival rates (like 27%)

But still, I am not getting any points that shadow the secondary island

Turns out the survival rate was related to not properly bounding the relevant islands, i.e. I was generating points with orbits inside of the main stable island and its companion 3-periodic islands. Now I have something which is much reasonable: 0.5% survival rate for t = pi, v \in (1.1, 1.2).

Even here, the points whose iteration survive are all points that orbit close to the main island during most of the orbit. None are found that orbit close to the second island. This doesn't mean those orbits don't exist, but rather they aren't quite as common for the starting region that I have been working with.

- escape time landscape around the stable manifold of my hyperbolic point.

