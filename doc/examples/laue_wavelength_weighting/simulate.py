"""
Compare the weighting of the wavelength term of the Laue refinement target on
simulated pink-beam stills.

A crystal is simulated on a flat detector in a polychromatic beam, with
Gaussian noise on the spot centroids. The starting model is misoriented and
its cell perturbed, and the wavelengths of the reflections are assigned (as
the LaueAssigner of dials.index does, lambda_obs = 2 d sin(theta_obs)) with
either the true or the starting cell. The crystal is then refined with each
setting of refinement.reflections.weighting_strategy.wavelength_variance, and
the refined orientation and cell are compared with the true ones.

Usage:
    dials.python simulate.py [--seeds 5] [--output results.json]
    dials.python simulate.py --paired 20 [--output paired.json]

The first form runs every scenario and mode and prints the mean and standard
deviation over the seeds of the misorientation (mrad), the largest relative
error of a, b and c (%), the positional RMSD (px) and the median standard
deviation of the assigned wavelengths in the weights (A). The second compares
selected modes with dropping the wavelength term (wavelength_weight=0) on the
same data for each seed, and prints the mean paired difference in
misorientation, its standard error and the number of seeds on which the mode
was better.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random

import numpy as np

import libtbx.phil
from cctbx import uctbx
from dxtbx.model import BeamFactory, Crystal, DetectorFactory, Experiment
from dxtbx.model.experiment_list import ExperimentList
from scitbx import matrix

from dials.algorithms.refinement.refiner import RefinerFactory
from dials.algorithms.refinement.refiner import phil_scope as refiner_phil
from dials.algorithms.refinement.weighting_strategies import (
    laue_assigned_wavelength_variances,
)
from dials.array_family import flex

WAVELENGTH_RANGE = (0.97, 1.15)  # A
PIXEL_SIZE = 0.1  # mm
SIGMA_PX = 0.5  # centroid noise
MISORIENTATION = 0.003  # rad about each of x, y and z
CELL_PERTURBATION = (1.0, 1.003, 0.997)  # scale of a, b and c in the start


def make_experiments():
    """A P1 crystal (40, 50, 60 A) on a 200 x 200 mm detector at 150 mm"""
    beam = BeamFactory.make_polychromatic_beam(
        direction=(0, 0, 1), wavelength_range=WAVELENGTH_RANGE
    )
    detector = DetectorFactory.simple(
        sensor="PAD",
        distance=150,
        beam_centre=(100, 100),
        fast_direction="+x",
        slow_direction="-y",
        pixel_size=(PIXEL_SIZE, PIXEL_SIZE),
        image_size=(2000, 2000),
    )
    crystal = Crystal(
        real_space_a=(40, 0, 0),
        real_space_b=(0, 50, 0),
        real_space_c=(0, 0, 60),
        space_group_symbol="P 1",
    )
    R = matrix.col((1, 2, 3)).normalize().axis_and_angle_as_r3_rotation_matrix(0.5)
    crystal.set_U(R)
    experiments = ExperimentList()
    experiments.append(Experiment(beam=beam, detector=detector, crystal=crystal))
    return experiments


def simulate(experiments, d_min=2.0, sigma_px=SIGMA_PX, seed=0):
    """
    The reflections that diffract within the spectrum and hit the detector,
    with Gaussian noise on the centroids. The wavelengths are assigned from the
    noisy positions with the true cell.
    """
    rng = random.Random(seed)
    expt = experiments[0]
    panel = expt.detector[0]
    unit_s0 = matrix.col(expt.beam.get_unit_s0())
    UB = matrix.sqr(expt.crystal.get_A())
    uc = expt.crystal.get_unit_cell()
    hmax = [int(m) for m in uc.max_miller_indices(d_min)]

    rows = []
    for h in range(-hmax[0], hmax[0] + 1):
        for k in range(-hmax[1], hmax[1] + 1):
            for l in range(-hmax[2], hmax[2] + 1):
                if (h, k, l) == (0, 0, 0) or uc.d((h, k, l)) < d_min:
                    continue
                q = UB * matrix.col((h, k, l))
                wavelength = -2 * unit_s0.dot(q) / q.length_sq()
                if not WAVELENGTH_RANGE[0] <= wavelength <= WAVELENGTH_RANGE[1]:
                    continue
                try:
                    x, y = panel.get_ray_intersection(unit_s0 / wavelength + q)
                except RuntimeError:
                    continue
                if panel.is_coord_valid_mm((x, y)):
                    rows.append(((h, k, l), x, y))

    sigma_mm = sigma_px * PIXEL_SIZE
    n = len(rows)
    refl = flex.reflection_table()
    refl["miller_index"] = flex.miller_index([r[0] for r in rows])
    refl["id"] = flex.int(n, 0)
    refl["panel"] = flex.size_t(n, 0)
    refl["xyzobs.mm.value"] = flex.vec3_double(
        [
            (x + rng.gauss(0, sigma_mm), y + rng.gauss(0, sigma_mm), 0)
            for _, x, y in rows
        ]
    )
    refl["xyzobs.mm.variance"] = flex.vec3_double(n, (sigma_mm**2, sigma_mm**2, 0))
    refl.set_flags(flex.bool(n, True), refl.flags.indexed)
    assign_wavelengths(experiments, refl)
    return refl


def assign_wavelengths(experiments, refl):
    """
    Assign the wavelengths with the crystal model of the experiment, as the
    LaueAssigner does: lambda = |s1/|s1| - s0/|s0|| d.
    """
    expt = experiments[0]
    panel = expt.detector[0]
    unit_s0 = matrix.col(expt.beam.get_unit_s0())
    uc = expt.crystal.get_unit_cell()
    w = flex.double()
    s0 = flex.vec3_double()
    s1 = flex.vec3_double()
    for h, (x, y, _) in zip(refl["miller_index"], refl["xyzobs.mm.value"]):
        u1 = matrix.col(panel.get_lab_coord((x, y))).normalize()
        wavelength = (u1 - unit_s0).length() * uc.d(h)
        w.append(wavelength)
        s0.append(unit_s0 / wavelength)
        s1.append(u1 / wavelength)
    refl["wavelength"] = w
    refl["s0"] = s0
    refl["s1"] = s1
    refl["rlp"] = s1 - s0


def b_covariance(crystal, rel_sigma, n=4000, seed=0):
    """
    The covariance of the elements of B for independent relative errors
    rel_sigma in a, b and c (angles fixed), sampled.
    """
    rng = np.random.default_rng(seed)
    a, b, c, al, be, ga = crystal.get_unit_cell().parameters()
    samples = []
    for _ in range(n):
        f = 1 + rng.normal(0, rel_sigma, 3)
        uc = uctbx.unit_cell((a * f[0], b * f[1], c * f[2], al, be, ga))
        samples.append(np.array(uc.fractionalization_matrix()).reshape(3, 3).T.ravel())
    return np.cov(np.array(samples).T)


def run(
    seed,
    mode="weight",
    wavelength_weight=0,
    assign_with="start",
    cell_fix="a",
    cell_sigma=0.0,
    bcov_sigma=None,
    cell_error="independent",
    prior_sigma=None,
):
    """
    Refine one simulated still with one setting, and return the errors.

    Args:
        seed: The seed of the centroid noise.
        mode: wavelength_variance (weight, assignment or spectrum).
        wavelength_weight: The wavelength_weight for mode=weight.
        assign_with: Assign the wavelengths with the "true" or "start" cell.
        cell_fix: "a" to fix the a-axis parameter (as the Laue defaults
            do), "none" to refine the whole cell.
        cell_sigma: assignment.cell_relative_sigma.
        bcov_sigma: If set, give the starting crystal a B covariance for
            this relative error of a, b and c and use it.
        cell_error: assignment.cell_error (independent or correlated).
        prior_sigma: If set, restrain a, b and c to the starting cell with
            this relative standard deviation (tie_to_target).
    """
    experiments = make_experiments()
    refl = simulate(experiments, seed=seed)
    crystal = experiments[0].crystal
    U_true = matrix.sqr(crystal.get_U())
    cell_true = crystal.get_unit_cell().parameters()

    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        R = matrix.col(axis).axis_and_angle_as_r3_rotation_matrix(MISORIENTATION)
        crystal.set_U(R * matrix.sqr(crystal.get_U()))
    B = matrix.sqr(crystal.get_B())
    crystal.set_B(matrix.diag([1 / f for f in CELL_PERTURBATION]) * B)
    if assign_with == "start":
        assign_wavelengths(experiments, refl)
    if bcov_sigma:
        crystal.set_B_covariance(
            flex.double(b_covariance(crystal, bcov_sigma).ravel().tolist())
        )
    sigma_wavelength = flex.median(
        flex.sqrt(
            laue_assigned_wavelength_variances(
                experiments,
                refl,
                cell_relative_sigma=0.0 if cell_error == "correlated" else cell_sigma,
                use_B_covariance=bool(bcov_sigma),
            )
        )
    )

    restraint = ""
    if prior_sigma:
        start = crystal.get_unit_cell().parameters()
        values = ",".join(f"{v:.6f}" for v in start)
        sigmas = ",".join(f"{v * prior_sigma:.6f}" for v in start[:3]) + ",0,0,0"
        restraint = (
            "crystal.unit_cell.restraints.tie_to_target "
            f"{{ values={values}\n sigmas={sigmas} }}"
        )
    fix_list = "g_param_0" if cell_fix == "a" else "None"
    params = refiner_phil.fetch(
        source=libtbx.phil.parse(
            f"""
refinement.parameterisation {{
  beam.fix = all
  detector.fix = all
  crystal.unit_cell.fix_list = {fix_list}
  spherical_relp_model = True
  {restraint}
}}
refinement.reflections {{
  outlier.algorithm = null
  weighting_strategy {{
    override = statistical
    wavelength_weight = {wavelength_weight}
    wavelength_variance = {mode}
    assignment.cell_relative_sigma = {cell_sigma}
    assignment.use_B_covariance = {bool(bcov_sigma)}
    assignment.cell_error = {cell_error}
  }}
}}"""
        )
    ).extract()
    refiner = RefinerFactory.from_parameters_data_experiments(params, refl, experiments)
    refiner.run()

    refined = refiner.get_experiments()[0].crystal
    dU = matrix.sqr(refined.get_U()) * U_true.transpose()
    misorientation = math.acos(max(-1, min(1, (dU.trace() - 1) / 2))) * 1e3
    cell = refined.get_unit_cell().parameters()
    cell_error_pct = max(abs(x - t) / t for x, t in zip(cell[:3], cell_true[:3])) * 100
    rmsd_x, rmsd_y, _ = refiner.rmsds()
    return {
        "misorientation_mrad": misorientation,
        "cell_error_pct": cell_error_pct,
        "rmsd_xy_px": (rmsd_x + rmsd_y) / 2 / PIXEL_SIZE,
        "sigma_wavelength_A": sigma_wavelength,
    }


BASE_MODES = [
    ("weight, 1e4", {"mode": "weight", "wavelength_weight": 1e4}),
    ("weight, 100", {"mode": "weight", "wavelength_weight": 100}),
    ("weight, 0", {"mode": "weight", "wavelength_weight": 0}),
    ("assignment", {"mode": "assignment"}),
    ("spectrum", {"mode": "spectrum"}),
]
INDEPENDENT_CELL_MODES = [
    (
        f"assignment, independent cell σ {s * 100:g}%",
        {"mode": "assignment", "cell_sigma": s},
    )
    for s in (0.001, 0.003, 0.01, 0.03)
] + [
    (
        "assignment, independent B covariance (σ 0.3%)",
        {"mode": "assignment", "bcov_sigma": 0.003},
    )
]
CORRELATED_CELL_MODES = [
    (
        "assignment, correlated, no prior",
        {"mode": "assignment", "cell_error": "correlated"},
    )
] + [
    (
        f"assignment, correlated, prior σ {s * 100:g}%",
        {"mode": "assignment", "cell_error": "correlated", "prior_sigma": s},
    )
    for s in (0.0001, 0.001, 0.003, 0.01)
]

SCENARIOS = {
    # name: (assign_with, cell_fix, modes)
    "true_cell_partly_fixed": (
        "true",
        "a",
        BASE_MODES + INDEPENDENT_CELL_MODES[1:2] + CORRELATED_CELL_MODES,
    ),
    "start_cell_partly_fixed": (
        "start",
        "a",
        BASE_MODES + INDEPENDENT_CELL_MODES + CORRELATED_CELL_MODES,
    ),
    "start_cell_free": (
        "start",
        "none",
        BASE_MODES[1:] + INDEPENDENT_CELL_MODES + CORRELATED_CELL_MODES,
    ),
}

# (name, assign_with, mode settings), each compared with weight, 0
PAIRED = [
    ("true cell, assignment", "true", {"mode": "assignment"}),
    (
        "true cell, assignment, independent cell σ 0.3%",
        "true",
        {"mode": "assignment", "cell_sigma": 0.003},
    ),
    (
        "start cell, assignment, independent cell σ 0.3%",
        "start",
        {"mode": "assignment", "cell_sigma": 0.003},
    ),
    (
        "start cell, assignment, independent cell σ 1%",
        "start",
        {"mode": "assignment", "cell_sigma": 0.01},
    ),
    (
        "start cell, assignment, correlated, no prior",
        "start",
        {"mode": "assignment", "cell_error": "correlated"},
    ),
    (
        "start cell, assignment, correlated, prior σ 0.3%",
        "start",
        {"mode": "assignment", "cell_error": "correlated", "prior_sigma": 0.003},
    ),
]


def summarise(label, seeds, **kwargs):
    """
    The mean and standard deviation over the seeds of the errors of one
    setting. Runs where refinement fails (for example, when the cell scale is
    undetermined and the minimiser leaves the space of valid cells) are
    counted and left out.
    """
    results = []
    failed = 0
    for seed in seeds:
        try:
            results.append(run(seed, **kwargs))
        except RuntimeError:
            failed += 1
    summary = {"label": label, "failed": failed, "n": len(seeds)}
    if not results:
        print(f"  {label:48s} failed for all {failed} seeds", flush=True)
        return summary
    for key in results[0]:
        values = np.array([r[key] for r in results])
        summary[key] = float(values.mean())
        summary[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    print(
        f"  {label:48s} "
        f"misorientation {summary['misorientation_mrad']:.3f} "
        f"± {summary['misorientation_mrad_sd']:.3f} mrad, "
        f"cell error {summary['cell_error_pct']:.4f} "
        f"± {summary['cell_error_pct_sd']:.4f} %, "
        f"RMSD {summary['rmsd_xy_px']:.3f} px, "
        f"σλ {summary['sigma_wavelength_A']:.2e} A"
        + (f" ({failed} of {len(seeds)} failed)" if failed else ""),
        flush=True,
    )
    return summary


def run_scenarios(seeds):
    results = {}
    for name, (assign_with, cell_fix, modes) in SCENARIOS.items():
        print(name, flush=True)
        results[name] = [
            summarise(label, seeds, assign_with=assign_with, cell_fix=cell_fix, **kw)
            for label, kw in modes
        ]
    return results


def run_paired(n_seeds):
    results = {}
    for name, assign_with, kwargs in PAIRED:
        diffs = np.array(
            [
                run(seed, assign_with=assign_with, **kwargs)["misorientation_mrad"]
                - run(seed, assign_with=assign_with, mode="weight")[
                    "misorientation_mrad"
                ]
                for seed in range(n_seeds)
            ]
        )
        se = float(diffs.std(ddof=1) / math.sqrt(n_seeds))
        better = int((diffs < 0).sum())
        print(
            f"  {name:52s} mean difference {diffs.mean():+.3f} mrad, "
            f"SE {se:.3f}, better in {better}/{n_seeds}",
            flush=True,
        )
        results[name] = {"mean": float(diffs.mean()), "se": se, "better": better}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--paired", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    if args.paired:
        results = run_paired(args.paired)
    else:
        results = run_scenarios(range(args.seeds))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
