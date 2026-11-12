# Laue wavelength term weighting: simulations

`simulate.py` compares the settings of
`refinement.reflections.weighting_strategy.wavelength_variance` and its options on simulated pink-beam Laue stills.

## The simulated still

- A P1 crystal, 40 × 50 × 60 Å, with d_min = 2 Å.
- A 0.97–1.15 Å spectrum.
- A 200 × 200 mm detector at 150 mm, with 0.1 mm pixels.
- 1045 spots with 0.5 px Gaussian centroid noise.

## Starting model and sets

Refinement starts from a crystal misoriented by 3 mrad about each axis, with the cell from indexing. The wavelengths are assigned with that same cell and with the starting detector geometry, as the LaueAssigner in `dials.index` does: λ = |ŝ1 − ŝ0| d.

There are four sets:

| Set | Indexed cell | Detector distance |
|---|---|---|
| True Cell, distance fixed | true cell | fixed at the true value |
| True Cell, distance refined | true cell | refined from 0.5 mm too long |
| Cell with Error, distance fixed | b +0.3%, c −0.3% | fixed at the true value |
| Cell with Error, distance refined | b +0.3%, c −0.3% | refined from 0.5 mm too long |

At least two of the unit cell, detector distance and beam wavelength are always fixed. The a-axis of the cell is always fixed. The beam wavelength (the spectrum that determines the reflections) is always fixed too, because a Laue beam has no wavelength parameter in refinement.

## Refinement parameters common to all simulations

Every run uses the same `dials.refine` parameters, apart from the setting under test:

```
refinement {
  parameterisation {
    beam.fix = all
    detector.fix = all                      # distance fixed sets
    # detector.fix = orientation            # distance refined sets: only Dist
    # detector.fix_list = Shift1,Shift2     #   is refined
    crystal.unit_cell.fix_list = g_param_0  # a-axis fixed
    spherical_relp_model = True
  }
  reflections {
    outlier.algorithm = null
    weighting_strategy.override = statistical
  }
}
```

Everything else is the DIALS default:
- the LevMar engine, with at most 100 iterations;
- `target.bin_size_fraction = 0`, so refinement never stops early on an RMSD target and runs to convergence;
- all reflections used, with no subsampling;
- no restraints, except the `tie_to_target` cell prior in the correlated-model rows;
- scan-static models;
- no auto-reduction, since every run has far more reflections than parameters.

The refined parameters are therefore:
- the crystal orientation (Phi1–Phi3);
- the five free cell parameters (g_param_1–g_param_5);
- in the distance-refined sets, also the detector distance.

The beam direction, the detector position in its plane and the detector orientation are fixed at their true values.

These parameters differ from the Laue-stills defaults that `dials.refine` applies (`refine_defaults` in `dials/util/laue.py`) in two ways:
- **Outlier rejection.** The defaults reject outliers per image with MCD (minimum covariance determinant); here it is off, because the simulated data have no outliers.
- **Detector.** The defaults fix only the distance and refine the rest of the detector; here the whole detector is fixed, or only the distance is refined.

The beam is fixed in both. The defaults fix the two beam direction parameters, and a Laue beam has no wavelength parameter.

The simulations isolate the wavelength term, so the absolute errors are smaller than they would be with the detector also refined.

## Running

Run with `dials.python`:

```bash
dials.python simulate.py --seeds 5 --output results.json
```

```bash
dials.python simulate.py --paired 20 --output paired.json
```

The first command runs every set and setting. For each one it prints the mean and standard deviation over the seeds of:
- the misorientation (mrad);
- the largest relative error in a, b and c (%);
- the detector distance error (mm);
- the positional RMSD (px);
- the median σ(λ) in the weights.

The second command uses the fixed-distance sets. It compares selected settings with dropping the wavelength term (`wavelength_weight=0`) on identical data for each seed, and prints the mean paired difference in misorientation, its standard error, and how many seeds the setting was better on.

Runs where refinement fails are counted and left out of the summary.
