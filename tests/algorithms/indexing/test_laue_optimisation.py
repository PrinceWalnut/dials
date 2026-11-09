from __future__ import annotations

import numpy as np
import pytest

import libtbx
from cctbx import sgtbx, uctbx
from dxtbx.model import (
    BeamFactory,
    Crystal,
    DetectorFactory,
    Experiment,
    ExperimentList,
)
from scitbx import matrix

from dials.algorithms.indexing.laue_optimisation import (
    LaueAssigner,
    assign_laue_indices,
    auto_d_min,
    optimise_laue_indexing,
    phil_scope,
)
from dials.array_family import flex

WAVELENGTH_RANGE = (0.97, 1.15)
PEAK_WAVELENGTH = 1.04
D_MIN = 2.5


def _models():
    beam = BeamFactory.make_polychromatic_beam(
        direction=(0, 0, 1),
        wavelength_range=WAVELENGTH_RANGE,
        peak_wavelength=PEAK_WAVELENGTH,
    )
    detector = DetectorFactory.simple(
        "PAD", 150, (100, 100), "+x", "-y", (0.1, 0.1), (2000, 2000)
    )
    axes = matrix.sqr(
        uctbx.unit_cell((34.3, 45.6, 99.0, 90, 90, 90)).orthogonalization_matrix()
    ).transpose()
    crystal = Crystal(
        axes[0:3],
        axes[3:6],
        axes[6:9],
        space_group=sgtbx.space_group_info("P 21 21 21").group(),
    )
    U = (
        matrix.col((1, 2, 3))
        .normalize()
        .axis_and_angle_as_r3_rotation_matrix(40, deg=True)
    )
    crystal.set_U(U)
    return beam, detector, crystal


def _simulated_spots(beam, detector, crystal):
    """
    Miller indices, wavelengths and observed mm centroids of the reflections in
    the beam spectrum that hit the detector, leaving out harmonics.
    """
    assigner = LaueAssigner(
        beam.get_unit_s0(),
        np.array([[0.0, 0.0, -1.0]]),
        crystal,
        *WAVELENGTH_RANGE,
        PEAK_WAVELENGTH,
        D_MIN,
    )
    H = assigner.Hall
    q = H @ np.array(crystal.get_A()).reshape(3, 3).T
    s0 = np.array(beam.get_unit_s0())
    wavelength = -2.0 * (q @ s0) / (q * q).sum(-1)
    in_spectrum = (wavelength > WAVELENGTH_RANGE[0]) & (
        wavelength < WAVELENGTH_RANGE[1]
    )
    H, q, wavelength = H[in_spectrum], q[in_spectrum], wavelength[in_spectrum]
    rays = H // np.gcd.reduce(H, axis=-1)[:, None]
    _, inverse, counts = np.unique(
        rays, axis=0, return_inverse=True, return_counts=True
    )
    single = counts[inverse.ravel()] == 1
    H, q, wavelength = H[single], q[single], wavelength[single]

    panel = detector[0]
    hkl, wavelengths, xy = [], [], []
    for h, qi, w in zip(H, q, wavelength):
        s1 = matrix.col(qi + s0 / w)
        try:
            mm = panel.get_ray_intersection(s1)
        except RuntimeError:
            continue
        px = panel.millimeter_to_pixel(mm)
        if not (0 < px[0] < 2000 and 0 < px[1] < 2000):
            continue
        hkl.append(tuple(int(i) for i in h))
        wavelengths.append(w)
        xy.append(mm)
    return hkl, np.array(wavelengths), xy


def _reflections(xy, ids):
    reflections = flex.reflection_table()
    n = len(xy)
    reflections["id"] = flex.int(ids)
    reflections["imageset_id"] = flex.int(n, 0)
    reflections["panel"] = flex.size_t(n, 0)
    reflections["xyzobs.mm.value"] = flex.vec3_double([(x, y, 0) for x, y in xy])
    reflections["miller_index"] = flex.miller_index(n, (0, 0, 0))
    reflections["wavelength"] = flex.double(n, PEAK_WAVELENGTH)
    reflections["s0"] = flex.vec3_double(n, (0, 0, -1 / PEAK_WAVELENGTH))
    reflections["flags"] = flex.size_t(n, 0)
    return reflections


@pytest.mark.parametrize("misset_deg", [0.0, 0.3])
def test_optimise_laue_indexing(misset_deg):
    beam, detector, crystal = _models()
    hkl, wavelengths, xy = _simulated_spots(beam, detector, crystal)
    assert len(hkl) > 100

    # Index half of the spots to start with, and misset the crystal
    ids = [0 if i % 2 else -1 for i in range(len(hkl))]
    reflections = _reflections(xy, ids)
    indexed_crystal = Crystal(crystal)
    misset = (
        matrix.col((0, 1, 1))
        .normalize()
        .axis_and_angle_as_r3_rotation_matrix(misset_deg, deg=True)
    )
    indexed_crystal.set_U(misset * matrix.sqr(crystal.get_U()))
    experiments = ExperimentList(
        [Experiment(beam=beam, detector=detector, crystal=indexed_crystal)]
    )
    imported = ExperimentList([Experiment(beam=beam, detector=detector)])
    params = phil_scope.extract()
    params.d_min = D_MIN

    optimised = optimise_laue_indexing(experiments, reflections, imported, params)

    # All spots, including those not indexed before, are assigned correctly
    assert list(optimised["id"]) == [0] * len(hkl)
    assert list(optimised["miller_index"]) == hkl
    assert optimised.get_flags(optimised.flags.indexed).all_eq(True)
    assert list(optimised["wavelength"]) == pytest.approx(list(wavelengths))
    # The crystal orientation is recovered
    assert indexed_crystal.get_U() == pytest.approx(crystal.get_U(), abs=1e-6)

    # s0, s1 and rlp are consistent with the wavelengths
    w = optimised["wavelength"].as_numpy_array()
    s0 = np.array(optimised["s0"])
    s1 = np.array(optimised["s1"])
    assert np.linalg.norm(s0, axis=1) * w == pytest.approx(np.ones(len(w)))
    assert np.linalg.norm(s1, axis=1) * w == pytest.approx(np.ones(len(w)))
    assert np.array(optimised["rlp"]) == pytest.approx(s1 - s0)


def test_optimise_laue_indexing_unassigned():
    beam, detector, crystal = _models()
    hkl, _, xy = _simulated_spots(beam, detector, crystal)
    # A narrower spectrum than the data, so that some spots are not assigned
    params = phil_scope.extract()
    params.d_min = D_MIN
    params.n_macrocycles = 0
    params.wavelength_min = 1.0
    params.wavelength_max = 1.1
    reflections = _reflections(xy, [0] * len(hkl))
    experiments = ExperimentList(
        [Experiment(beam=beam, detector=detector, crystal=Crystal(crystal))]
    )
    imported = ExperimentList([Experiment(beam=beam, detector=detector)])

    optimised = optimise_laue_indexing(experiments, reflections, imported, params)

    indexed = optimised["id"] == 0
    assert 0 < indexed.count(True) < len(hkl)
    assert optimised.get_flags(optimised.flags.indexed) == indexed
    w = optimised["wavelength"]
    assert flex.min(w.select(indexed)) >= 1.0
    assert flex.max(w.select(indexed)) <= 1.1
    # Unassigned spots are unindexed, with the peak wavelength of the beam
    unindexed = optimised.select(~indexed)
    assert set(unindexed["id"]) == {-1}
    assert set(unindexed["miller_index"]) == {(0, 0, 0)}
    assert list(unindexed["wavelength"]) == pytest.approx(
        [PEAK_WAVELENGTH] * len(unindexed)
    )


def test_auto_d_min():
    unit_s0 = (0, 0, -1)
    two_theta = np.radians([10.0, 30.0, 48.0])
    s1 = np.stack(
        [np.sin(two_theta), np.zeros(3), -np.cos(two_theta)], axis=-1
    ) * np.array([[1], [2], [3]])
    assert auto_d_min(unit_s0, s1, 0.97) == pytest.approx(
        0.97 / (2 * np.sin(np.radians(24.0)))
    )


def test_optimise_laue_indexing_auto_d_min():
    beam, detector, crystal = _models()
    hkl, wavelengths, xy = _simulated_spots(beam, detector, crystal)
    reflections = _reflections(xy, [0] * len(hkl))
    experiments = ExperimentList(
        [Experiment(beam=beam, detector=detector, crystal=Crystal(crystal))]
    )
    imported = ExperimentList([Experiment(beam=beam, detector=detector)])
    params = phil_scope.extract()
    assert params.d_min is libtbx.Auto

    optimised = optimise_laue_indexing(experiments, reflections, imported, params)

    # The resolution limit is no higher than that of any spot, so every spot is
    # assigned, with the wavelength of its reflection (or of a harmonic of it)
    assert list(optimised["id"]) == [0] * len(hkl)
    assert optimised.get_flags(optimised.flags.indexed).all_eq(True)
    for h, h_assigned, w, w_assigned in zip(
        hkl, optimised["miller_index"], wavelengths, optimised["wavelength"]
    ):
        n = np.gcd.reduce(h_assigned) / np.gcd.reduce(h)
        assert np.array(h_assigned) == pytest.approx(n * np.array(h))
        assert w_assigned == pytest.approx(w / n)


def test_optimise_laue_indexing_nothing_indexed():
    beam, detector, _ = _models()
    imported = ExperimentList([Experiment(beam=beam, detector=detector)])
    reflections = flex.reflection_table()
    optimised = optimise_laue_indexing(
        ExperimentList(), reflections, imported, phil_scope.extract()
    )
    assert len(optimised) == 0


def test_assign_laue_indices():
    beam, detector, crystal = _models()
    hkl, wavelengths, xy = _simulated_spots(beam, detector, crystal)
    reflections = _reflections(xy, [-1] * len(hkl))
    U = crystal.get_U()
    experiment = Experiment(beam=beam, detector=detector, crystal=crystal)
    params = phil_scope.extract()
    params.d_min = D_MIN

    assigned = assign_laue_indices(experiment, reflections, params)

    assert list(assigned["id"]) == [0] * len(hkl)
    assert assigned.get_flags(assigned.flags.indexed).all_eq(True)
    assert list(assigned["miller_index"]) == hkl
    assert list(assigned["wavelength"]) == pytest.approx(list(wavelengths))
    w = assigned["wavelength"].as_numpy_array()
    s0 = np.array(assigned["s0"])
    s1 = np.array(assigned["s1"])
    assert np.linalg.norm(s0, axis=1) * w == pytest.approx(np.ones(len(w)))
    assert np.linalg.norm(s1, axis=1) * w == pytest.approx(np.ones(len(w)))
    assert np.array(assigned["rlp"]) == pytest.approx(s1 - s0)
    # The crystal model is not changed
    assert crystal.get_U() == U
