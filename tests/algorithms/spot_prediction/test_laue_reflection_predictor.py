from __future__ import annotations

from cctbx import sgtbx, uctbx
from dxtbx.model import BeamFactory, Crystal, DetectorFactory, Experiment
from scitbx import matrix

from dials.algorithms.spot_prediction import LaueReflectionPredictor
from dials.array_family import flex


def _experiment():
    beam = BeamFactory.make_polychromatic_beam(
        direction=(0, 0, 1), wavelength_range=(0.97, 1.15), peak_wavelength=1.04
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
    crystal.set_U(
        matrix.col((1, 2, 3))
        .normalize()
        .axis_and_angle_as_r3_rotation_matrix(40, deg=True)
    )
    return Experiment(beam=beam, detector=detector, crystal=crystal)


def test_for_reflection_table_keeps_unpredictable_reflections():
    experiment = _experiment()
    predictor = LaueReflectionPredictor(experiment, 2.0)
    A = experiment.crystal.get_A()
    predicted = predictor.for_ub(A)
    assert len(predicted) > 0

    # A reflection that is predicted on the detector, and its Friedel mate,
    # whose ray (at a negative wavelength) does not meet the detector
    h = predicted["miller_index"][0]
    reflections = flex.reflection_table()
    reflections["miller_index"] = flex.miller_index([h, tuple(-i for i in h)])
    reflections["panel"] = flex.size_t(2, 0)
    reflections["flags"] = flex.size_t(2, 0)
    reflections["id"] = flex.int(2, 0)

    predictor.for_reflection_table(reflections, A)

    assert len(reflections) == 2
    assert list(reflections.get_flags(reflections.flags.predicted)) == [True, False]
    assert reflections["xyzcal.px"][0] == predicted["xyzcal.px"][0]
    assert reflections["xyzcal.px"][1] == (0, 0, 0)
    assert list(reflections["miller_index"]) == [h, tuple(-i for i in h)]
