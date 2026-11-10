# Laue wavelength term weighting: simulations

`simulate.py` compares the settings of
`refinement.reflections.weighting_strategy.wavelength_variance` (and the
`assignment` cell error models) on simulated pink-beam Laue stills.

The simulated still has these properties:
- a P1 crystal, 40 × 50 × 60 Å, with d_min = 2 Å;
- a 0.97–1.15 Å spectrum;
- a 200 × 200 mm detector at 150 mm, with 0.1 mm pixels;
- 1045 spots with 0.5 px Gaussian centroid noise.

The starting model is misoriented by 3 mrad about each axis, with b scaled by +0.3% and c by −0.3%. The detector and beam are fixed in refinement. Wavelengths are assigned as the LaueAssigner does, λ = |ŝ1 − ŝ0| d. The assignment uses either the true cell (the idealised case) or the starting cell (what happens in practice).

Run with `dials.python`:

```bash
dials.python simulate.py --seeds 5 --output results.json
```

```bash
dials.python simulate.py --paired 20 --output paired.json
```

The first command runs every scenario and setting. For each one it prints the mean and standard deviation over the seeds of:
- the misorientation (mrad);
- the largest relative error in a, b and c (%);
- the positional RMSD (px);
- the median σ(λ) in the weights.

The second command compares selected settings with dropping the wavelength term (`wavelength_weight=0`) on identical data for each seed. It prints the mean paired difference in misorientation, its standard error, and how many seeds the setting was better on.

Runs where refinement fails are counted and left out of the summary. This happens, for example, when nothing fixes the cell scale.
