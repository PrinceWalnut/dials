"""
Tests for the Laue wavelength variance models of the refinement target:
wavelength_variance=assignment (the variance of the wavelengths assigned in
indexing, propagated from the centroid variances) and
wavelength_variance=spectrum (maximum likelihood for a Gaussian spectrum).
"""

from __future__ import annotations

import math
import random

import pytest

import libtbx.phil
from dxtbx.model import BeamFactory, Crystal, DetectorFactory, Experiment
from dxtbx.model.experiment_list import ExperimentList
from scitbx import matrix

from dials.algorithms.refinement.weighting_strategies import (
    LaueWavelengthVarianceWeightingStrategy,
    laue_assigned_wavelength_variances,
)
from dials.array_family import flex

WAVELENGTH_RANGE = (0.97, 1.15)


def _make_experiments():
    beam = BeamFactory.make_polychromatic_beam(
        direction=(0, 0, 1), wavelength_range=WAVELENGTH_RANGE
    )
    detector = DetectorFactory.simple(
        sensor="PAD",
        distance=150,
        beam_centre=(100, 100),
        fast_direction="+x",
        slow_direction="-y",
        pixel_size=(0.1, 0.1),
        image_size=(2000, 2000),
    )
    crystal = Crystal(
        real_space_a=(40, 0, 0),
        real_space_b=(0, 50, 0),
        real_space_c=(0, 0, 60),
        space_group_symbol="P 1",
    )
    # A general orientation
    R = matrix.col((1, 2, 3)).normalize().axis_and_angle_as_r3_rotation_matrix(0.5)
    crystal.set_U(R)
    experiments = ExperimentList()
    experiments.append(Experiment(beam=beam, detector=detector, crystal=crystal))
    return experiments


def _simulate(experiments, d_min=2.0, sigma_px=0.3, seed=0):
    """Reflections at the positions where they diffract, with Gaussian noise
    on the centroids and the wavelengths assigned from the noisy positions, as
    the LaueAssigner assigns them"""
    rng = random.Random(seed)
    expt = experiments[0]
    beam, detector, crystal = expt.beam, expt.detector, expt.crystal
    panel = detector[0]
    unit_s0 = matrix.col(beam.get_unit_s0())
    UB = matrix.sqr(crystal.get_A())
    uc = crystal.get_unit_cell()
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
                s1 = unit_s0 / wavelength + q
                try:
                    x, y = panel.get_ray_intersection(s1)
                except RuntimeError:
                    continue
                if not panel.is_coord_valid_mm((x, y)):
                    continue
                rows.append(((h, k, l), x, y, q))
    sigma_mm = sigma_px * panel.get_pixel_size()[0]

    n = len(rows)
    refl = flex.reflection_table()
    refl["miller_index"] = flex.miller_index([r[0] for r in rows])
    refl["id"] = flex.int(n, 0)
    refl["panel"] = flex.size_t(n, 0)
    xyz = flex.vec3_double()
    wavelengths = flex.double()
    s0 = flex.vec3_double()
    s1 = flex.vec3_double()
    for _, x, y, q in rows:
        x += rng.gauss(0, sigma_mm)
        y += rng.gauss(0, sigma_mm)
        xyz.append((x, y, 0))
        # Assigned wavelength: |s1/|s1| - s0/|s0|| / |q|
        u1 = matrix.col(panel.get_lab_coord((x, y))).normalize()
        wavelength = (u1 - unit_s0).length() / q.length()
        wavelengths.append(wavelength)
        s0.append(unit_s0 / wavelength)
        s1.append(u1 / wavelength)
    refl["xyzobs.mm.value"] = xyz
    refl["xyzobs.mm.variance"] = flex.vec3_double(n, (sigma_mm**2, sigma_mm**2, 0))
    refl["wavelength"] = wavelengths
    refl["s0"] = s0
    refl["s1"] = s1
    refl["rlp"] = s1 - s0
    refl.set_flags(flex.bool(n, True), refl.flags.indexed)
    return refl


def test_assigned_wavelength_variances_match_finite_differences():
    experiments = _make_experiments()
    refl = _simulate(experiments, sigma_px=0.0)
    sigma_mm = 0.03
    refl["xyzobs.mm.variance"] = flex.vec3_double(
        len(refl), (sigma_mm**2, sigma_mm**2, 0)
    )
    variances = laue_assigned_wavelength_variances(experiments, refl)

    panel = experiments[0].detector[0]
    unit_s0 = matrix.col(experiments[0].beam.get_unit_s0())

    def assigned(x, y, d):
        u1 = matrix.col(panel.get_lab_coord((x, y))).normalize()
        return (u1 - unit_s0).length() * d

    uc = experiments[0].crystal.get_unit_cell()
    eps = 1e-4
    for i in range(0, len(refl), max(1, len(refl) // 20)):
        x, y, _ = refl["xyzobs.mm.value"][i]
        d = uc.d(refl["miller_index"][i])
        dl_dx = (assigned(x + eps, y, d) - assigned(x - eps, y, d)) / (2 * eps)
        dl_dy = (assigned(x, y + eps, d) - assigned(x, y - eps, d)) / (2 * eps)
        expected = (dl_dx**2 + dl_dy**2) * sigma_mm**2
        assert variances[i] == pytest.approx(expected, rel=1e-4)


def test_spectrum_weights_from_beam_and_reflections():
    experiments = _make_experiments()
    refl = _simulate(experiments)

    strategy = LaueWavelengthVarianceWeightingStrategy(experiments, model="spectrum")
    strategy.calculate_weights(refl)
    centre, sigma = strategy.spectra[0]
    assert centre == pytest.approx(sum(WAVELENGTH_RANGE) / 2)
    fwhm = WAVELENGTH_RANGE[1] - WAVELENGTH_RANGE[0]
    assert sigma == pytest.approx(fwhm / (2 * math.sqrt(2 * math.log(2))))
    assert set(refl["wavelength_obs"]) == {centre}
    _, _, wz = refl["xyzobs.mm.weights"].parts()
    assert list(wz) == pytest.approx([1 / sigma**2] * len(wz))

    strategy = LaueWavelengthVarianceWeightingStrategy(
        experiments, model="spectrum", spectrum_source="reflections"
    )
    strategy.calculate_weights(refl)
    centre, sigma = strategy.spectra[0]
    assert centre == pytest.approx(flex.median(refl["wavelength"]))
    assert 0 < sigma < fwhm

    strategy = LaueWavelengthVarianceWeightingStrategy(
        experiments, model="spectrum", spectrum_centre=1.0, spectrum_sigma=0.01
    )
    strategy.calculate_weights(refl)
    assert strategy.spectra[0] == (1.0, 0.01)


def test_assignment_weights():
    experiments = _make_experiments()
    refl = _simulate(experiments)
    strategy = LaueWavelengthVarianceWeightingStrategy(experiments, model="assignment")
    strategy.calculate_weights(refl)
    _, _, wz = refl["xyzobs.mm.weights"].parts()
    variances = laue_assigned_wavelength_variances(experiments, refl)
    assert list(wz) == pytest.approx(list(1 / variances))
    assert list(refl["wavelength_obs"]) == list(refl["wavelength"])


@pytest.mark.parametrize("wavelength_variance", ["weight", "assignment", "spectrum"])
def test_laue_stills_refinement(wavelength_variance):
    """Refine a misoriented crystal against simulated Laue stills data with
    each wavelength variance model"""
    from dials.algorithms.refinement.refiner import RefinerFactory
    from dials.algorithms.refinement.refiner import phil_scope as refiner_phil

    experiments = _make_experiments()
    refl = _simulate(experiments, sigma_px=0.3)
    crystal = experiments[0].crystal
    U_true = matrix.sqr(crystal.get_U())

    # Misorient the crystal by 2 mrad about each axis
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        R = matrix.col(axis).axis_and_angle_as_r3_rotation_matrix(0.002)
        crystal.set_U(R * matrix.sqr(crystal.get_U()))

    params = refiner_phil.fetch(
        source=libtbx.phil.parse(
            f"""
            refinement {{
              parameterisation {{
                beam.fix = all
                detector.fix = all
                crystal.unit_cell.fix_list = g_param_0
                spherical_relp_model = True
              }}
              reflections {{
                outlier.algorithm = null
                weighting_strategy {{
                  override = statistical
                  wavelength_weight = 100
                  wavelength_variance = {wavelength_variance}
                }}
              }}
            }}
            """
        )
    ).extract()
    refiner = RefinerFactory.from_parameters_data_experiments(params, refl, experiments)
    refiner.run()
    U = matrix.sqr(refiner.get_experiments()[0].crystal.get_U())
    # Residual misorientation angle (rad)
    dU = U * U_true.transpose()
    angle = math.acos(max(-1.0, min(1.0, (dU.trace() - 1) / 2)))
    rmsd_x, rmsd_y, _ = refiner.rmsds()
    print(wavelength_variance, len(refl), angle, refiner.rmsds())
    assert angle < 2e-4
    # The simulated centroid errors are 0.03 mm
    assert rmsd_x < 0.035 and rmsd_y < 0.035


def test_assigned_wavelength_variances_with_cell_uncertainty():
    experiments = _make_experiments()
    refl = _simulate(experiments, sigma_px=0.0)
    base = laue_assigned_wavelength_variances(experiments, refl)

    # An isotropic relative error of the d-spacings adds (lambda s)^2
    s = 0.003
    with_cell = laue_assigned_wavelength_variances(
        experiments, refl, cell_relative_sigma=s
    )
    expected = base + (refl["wavelength"] * s) ** 2
    assert list(with_cell) == pytest.approx(list(expected))

    # The B matrix covariance is propagated to the d-spacings
    crystal = experiments[0].crystal
    rng = random.Random(1)
    A = [rng.gauss(0, 1e-5) for _ in range(81)]
    cov = matrix.sqr(A) * matrix.sqr(A).transpose()
    crystal.set_B_covariance(cov.as_flex_double_matrix())
    with_cov = laue_assigned_wavelength_variances(
        experiments, refl, use_B_covariance=True
    )
    B = matrix.sqr(crystal.get_B())
    eps = 1e-7
    for i in range(0, len(refl), max(1, len(refl) // 10)):
        h = matrix.col(refl["miller_index"][i])
        q_len = (B * h).length()
        J = []
        for k in range(9):
            dB = [0.0] * 9
            dB[k] = eps
            Bp = B + matrix.sqr(dB)
            Bm = B - matrix.sqr(dB)
            J.append(((Bp * h).length() - (Bm * h).length()) / (2 * eps))
        J = matrix.row(J)
        rel_var_d = (J * cov * J.transpose())[0] / q_len**2
        expected = base[i] + refl["wavelength"][i] ** 2 * rel_var_d
        assert with_cov[i] == pytest.approx(expected, rel=1e-5)
    # and ignored unless requested
    assert list(laue_assigned_wavelength_variances(experiments, refl)) == pytest.approx(
        list(base)
    )


def _build_refiner(experiments, refl, phil_str):
    from dials.algorithms.refinement.refiner import RefinerFactory
    from dials.algorithms.refinement.refiner import phil_scope as refiner_phil

    params = refiner_phil.fetch(source=libtbx.phil.parse(phil_str)).extract()
    return RefinerFactory.from_parameters_data_experiments(params, refl, experiments)


def test_correlated_cell_error_wavelength_gradients():
    """With a correlated cell error, the observed wavelengths follow the cell,
    and the gradients of the wavelength residuals match finite differences"""
    experiments = _make_experiments()
    refl = _simulate(experiments, sigma_px=0.3)
    refiner = _build_refiner(
        experiments,
        refl,
        """
        refinement {
          parameterisation {
            beam.fix = all
            detector.fix = all
            spherical_relp_model = True
          }
          reflections {
            outlier.algorithm = null
            weighting_strategy {
              override = statistical
              wavelength_variance = assignment
              assignment.cell_error = correlated
            }
          }
        }
        """,
    )
    target = refiner._target
    pred_param = refiner._pred_param
    target.predict()
    target.update_matches(force=True)
    matches = target._matches
    n = len(matches)
    assert "wavelength_obs_factor" in matches
    # At the indexing model, the recalculated wavelengths are the assigned ones
    assert list(matches["wavelength_obs"]) == pytest.approx(list(matches["wavelength"]))

    _, jacobian, _ = target.compute_residuals_and_gradients()
    p0 = pred_param.get_param_vals()
    names = pred_param.get_param_names()

    def wavelength_residuals(p):
        pred_param.set_param_vals(p)
        target.predict()
        target.update_matches(force=True)
        residuals, _ = target.compute_residuals()
        return residuals[2 * n :]

    for k, name in enumerate(names):
        eps = 1e-6
        pp = list(p0)
        pp[k] += eps
        pm = list(p0)
        pm[k] -= eps
        fd = (wavelength_residuals(pp) - wavelength_residuals(pm)) / (2 * eps)
        analytic = jacobian.matrix_copy_column(k)[2 * n :]
        # Rotation about the beam does not change the wavelengths, so allow
        # for finite difference noise as well as a relative error
        tolerance = 1e-8 + 1e-4 * flex.max(flex.abs(fd))
        assert flex.max(flex.abs(analytic - fd)) < tolerance, name
    pred_param.set_param_vals(p0)
