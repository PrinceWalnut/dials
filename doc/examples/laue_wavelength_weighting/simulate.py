"""
Compare the weighting of the wavelength term of the Laue refinement target on
simulated pink-beam stills.

A crystal is simulated on a flat detector in a polychromatic beam, with
Gaussian noise on the spot centroids. Refinement starts from a misoriented
crystal with the cell from indexing, which is either the true cell or a cell
with error (b +0.3%, c -0.3%). The wavelengths of the reflections are
assigned with the indexed cell and the starting detector geometry, as the
LaueAssigner of dials.index does,
lambda_obs = |s1/|s1| - s0/|s0|| d. The crystal is then refined with each
setting of refinement.reflections.weighting_strategy.wavelength_variance and
its options, and the refined orientation, cell and detector distance are
compared with the true ones.

At least two of the unit cell, the detector distance and the beam wavelength
are always fixed. A Laue beam has no refined wavelength parameter, so the
beam wavelength (the spectrum that determines the reflections) is always
fixed, and so is the a-axis of the cell. The detector distance is either
fixed at its true value, or refined from a start 0.5 mm too long.

Usage:
    dials.python simulate.py [--seeds 5] [--output results.json]
    dials.python simulate.py --paired 20 [--output paired.json]

The first form runs every set and setting and prints the mean and standard
deviation over the seeds of the misorientation (mrad), the largest relative
error of a, b and c (%), the distance error (mm), the positional RMSD (px)
and the median standard deviation of the assigned wavelengths in the
weights (A). The second compares selected settings with dropping the
wavelength term (wavelength_weight=0) on the same data for each seed, and
prints the mean paired difference in misorientation, its standard error and
the number of seeds on which the setting was better.
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
CELL_ERROR = (1.0, 1.003, 0.997)  # scale of a, b and c of the indexed cell with error
PRIOR_ERROR = (1.0, 1.001, 0.999)  # scale of a, b and c of a slightly incorrect prior
DISTANCE_ERROR = 0.5  # mm, of the starting distance when it is refined


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
    indexed_cell="true",
    distance="fixed",
    mode="weight",
    wavelength_weight=0,
    cell_sigma=0.0,
    bcov_sigma=None,
    cell_error="independent",
    prior_cell=None,
    prior_sigma=None,
    spectrum_source="beam",
):
    """
    Refine one simulated still with one setting, and return the errors.

    Args:
        seed: The seed of the centroid noise.
        indexed_cell: The cell from indexing, used as the starting cell and to
            assign the wavelengths: "true" or "error" (CELL_ERROR).
        distance: "fixed" at the true distance, or "refined" from a start
            DISTANCE_ERROR too long.
        mode: wavelength_variance (weight, assignment or spectrum).
        wavelength_weight: The wavelength_weight for mode=weight.
        cell_sigma: assignment.cell_relative_sigma.
        bcov_sigma: If set, give the indexed crystal a B covariance for this
            relative error of a, b and c and use it.
        cell_error: assignment.cell_error (independent or correlated).
        prior_cell: If set, restrain a, b and c to the "true" cell or to a
            "slightly_incorrect" one (PRIOR_ERROR), with tie_to_target...
        prior_sigma: ...and this relative standard deviation.
        spectrum_source: spectrum.source (beam or reflections).
    """
    experiments = make_experiments()
    refl = simulate(experiments, seed=seed)
    crystal = experiments[0].crystal
    panel = experiments[0].detector[0]
    U_true = matrix.sqr(crystal.get_U())
    cell_true = crystal.get_unit_cell().parameters()
    distance_true = panel.get_distance()

    # The starting model: misoriented, with the indexed cell
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        R = matrix.col(axis).axis_and_angle_as_r3_rotation_matrix(MISORIENTATION)
        crystal.set_U(R * matrix.sqr(crystal.get_U()))
    if indexed_cell == "error":
        B = matrix.sqr(crystal.get_B())
        crystal.set_B(matrix.diag([1 / f for f in CELL_ERROR]) * B)
    if distance == "refined":
        # Move the detector back along its normal
        normal = matrix.col(panel.get_normal())
        origin = matrix.col(panel.get_origin()) + DISTANCE_ERROR * normal
        panel.set_frame(panel.get_fast_axis(), panel.get_slow_axis(), origin)
    # Assign the wavelengths with the starting (indexing) model: the indexed
    # cell and the starting detector distance
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
    if prior_cell:
        scale = (1, 1, 1) if prior_cell == "true" else PRIOR_ERROR
        target = [v * f for v, f in zip(cell_true[:3], scale)] + list(cell_true[3:])
        values = ",".join(f"{v:.6f}" for v in target)
        sigmas = ",".join(f"{v * prior_sigma:.6f}" for v in target[:3]) + ",0,0,0"
        restraint = (
            "crystal.unit_cell.restraints.tie_to_target "
            f"{{ values={values}\n sigmas={sigmas} }}"
        )
    if distance == "fixed":
        detector = "detector.fix = all"
    else:
        detector = "detector.fix = orientation\n  detector.fix_list = Shift1,Shift2"
    params = refiner_phil.fetch(
        source=libtbx.phil.parse(
            f"""
refinement.parameterisation {{
  beam.fix = all
  {detector}
  crystal.unit_cell.fix_list = g_param_0
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
    spectrum.source = {spectrum_source}
  }}
}}"""
        )
    ).extract()
    refiner = RefinerFactory.from_parameters_data_experiments(params, refl, experiments)
    refiner.run()

    refined = refiner.get_experiments()[0]
    dU = matrix.sqr(refined.crystal.get_U()) * U_true.transpose()
    misorientation = math.acos(max(-1, min(1, (dU.trace() - 1) / 2))) * 1e3
    cell = refined.crystal.get_unit_cell().parameters()
    cell_error_pct = max(abs(x - t) / t for x, t in zip(cell[:3], cell_true[:3])) * 100
    rmsd_x, rmsd_y, _ = refiner.rmsds()
    return {
        "misorientation_mrad": misorientation,
        "cell_error_pct": cell_error_pct,
        "distance_error_mm": refined.detector[0].get_distance() - distance_true,
        "rmsd_xy_px": (rmsd_x + rmsd_y) / 2 / PIXEL_SIZE,
        "sigma_wavelength_A": sigma_wavelength,
    }


def _pct(s):
    return f"{s * 100:g}%"


MODES = (
    [
        ("weight, 1e4", {"mode": "weight", "wavelength_weight": 1e4}),
        ("weight, 100", {"mode": "weight", "wavelength_weight": 100}),
        ("weight, 0", {"mode": "weight", "wavelength_weight": 0}),
        ("assignment, no cell error", {"mode": "assignment"}),
    ]
    + [
        (
            f"assignment, independent cell σ {_pct(s)}",
            {"mode": "assignment", "cell_sigma": s},
        )
        for s in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
    ]
    + [
        (
            f"assignment, independent B covariance σ {_pct(s)}",
            {"mode": "assignment", "bcov_sigma": s},
        )
        for s in (0.001, 0.003, 0.01)
    ]
    + [
        (
            "assignment, correlated, no prior",
            {"mode": "assignment", "cell_error": "correlated"},
        )
    ]
    + [
        (
            f"assignment, correlated, {label} prior σ {_pct(s)}",
            {
                "mode": "assignment",
                "cell_error": "correlated",
                "prior_cell": prior,
                "prior_sigma": s,
            },
        )
        for prior, label in (
            ("true", "true cell"),
            ("slightly_incorrect", "±0.1% cell"),
        )
        for s in (0.0001, 0.001)
    ]
    + [
        ("spectrum, source beam", {"mode": "spectrum", "spectrum_source": "beam"}),
        (
            "spectrum, source reflections",
            {"mode": "spectrum", "spectrum_source": "reflections"},
        ),
    ]
)

SETS = {
    # name: (indexed_cell, distance)
    "true_cell_distance_fixed": ("true", "fixed"),
    "true_cell_distance_refined": ("true", "refined"),
    "cell_with_error_distance_fixed": ("error", "fixed"),
    "cell_with_error_distance_refined": ("error", "refined"),
}

# Settings compared with weight, 0 on the same data, with the distance fixed
PAIRED = [
    ("weight, 100", {"mode": "weight", "wavelength_weight": 100}),
    ("assignment, no cell error", {"mode": "assignment"}),
    (
        "assignment, independent cell σ 0.3%",
        {"mode": "assignment", "cell_sigma": 0.003},
    ),
    (
        "assignment, correlated, no prior",
        {"mode": "assignment", "cell_error": "correlated"},
    ),
    ("spectrum, source beam", {"mode": "spectrum", "spectrum_source": "beam"}),
    (
        "spectrum, source reflections",
        {"mode": "spectrum", "spectrum_source": "reflections"},
    ),
]


def summarise(label, seeds, **kwargs):
    """
    The mean and standard deviation over the seeds of the errors of one
    setting. Runs where refinement fails are counted and left out.
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
        f"distance error {summary['distance_error_mm']:+.4f} "
        f"± {summary['distance_error_mm_sd']:.4f} mm, "
        f"RMSD {summary['rmsd_xy_px']:.3f} px, "
        f"σλ {summary['sigma_wavelength_A']:.2e} A"
        + (f" ({failed} of {len(seeds)} failed)" if failed else ""),
        flush=True,
    )
    return summary


def run_sets(seeds):
    results = {}
    for name, (indexed_cell, distance) in SETS.items():
        print(name, flush=True)
        results[name] = [
            summarise(label, seeds, indexed_cell=indexed_cell, distance=distance, **kw)
            for label, kw in MODES
        ]
    return results


def run_paired(n_seeds):
    results = {}
    for indexed_cell in ("true", "error"):
        print(f"indexed cell: {indexed_cell}", flush=True)
        for name, kwargs in PAIRED:
            diffs = np.array(
                [
                    run(seed, indexed_cell=indexed_cell, **kwargs)[
                        "misorientation_mrad"
                    ]
                    - run(seed, indexed_cell=indexed_cell)["misorientation_mrad"]
                    for seed in range(n_seeds)
                ]
            )
            se = float(diffs.std(ddof=1) / math.sqrt(n_seeds))
            better = int((diffs < 0).sum())
            print(
                f"  {name:40s} mean difference {diffs.mean():+.3f} mrad, "
                f"SE {se:.3f}, better in {better}/{n_seeds}",
                flush=True,
            )
            results[f"{indexed_cell}: {name}"] = {
                "mean": float(diffs.mean()),
                "se": se,
                "better": better,
            }
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
        results = run_sets(range(args.seeds))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
