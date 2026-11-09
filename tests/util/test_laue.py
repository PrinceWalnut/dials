from __future__ import annotations

import pytest

import libtbx
from cctbx import sgtbx, uctbx
from dxtbx.model import BeamFactory, Crystal, Experiment, ExperimentList, Scan
from scitbx import matrix

from dials.command_line.find_spots import working_phil as find_spots_phil
from dials.command_line.index import working_phil as index_phil
from dials.command_line.refine import working_phil as refine_phil
from dials.util import laue
from dials.util.options import ArgumentParser


def _laue_beam(peak_wavelength=1.04):
    return BeamFactory.make_polychromatic_beam(
        direction=(0, 0, 1),
        wavelength_range=(0.97, 1.15),
        peak_wavelength=peak_wavelength,
    )


def _monochromatic_beam():
    return BeamFactory.simple(wavelength=1.0)


def test_is_laue_stills():
    assert laue.is_laue_stills(
        ExperimentList([Experiment(beam=_laue_beam()) for _ in range(3)])
    )
    # No experiments
    assert not laue.is_laue_stills(ExperimentList())
    # Monochromatic stills
    assert not laue.is_laue_stills(
        ExperimentList([Experiment(beam=_monochromatic_beam())])
    )
    # A mixture of Laue and monochromatic stills
    assert not laue.is_laue_stills(
        ExperimentList(
            [Experiment(beam=_laue_beam()), Experiment(beam=_monochromatic_beam())]
        )
    )
    # Laue data with a scan, e.g. a sequence
    assert not laue.is_laue_stills(
        ExperimentList(
            [
                Experiment(
                    beam=_laue_beam(),
                    scan=Scan(image_range=(1, 1), oscillation=(0.0, 0.0)),
                )
            ]
        )
    )


def test_pink_indexer_wavelength_and_bandwidth():
    wavelength, bandwidth = laue.pink_indexer_wavelength_and_bandwidth(_laue_beam())
    assert wavelength == pytest.approx(1.04)
    assert bandwidth == pytest.approx(100 * 0.18 / 1.04)

    # Without a peak wavelength, the wavelength range midpoint is used
    wavelength, bandwidth = laue.pink_indexer_wavelength_and_bandwidth(
        _laue_beam(peak_wavelength=None)
    )
    assert wavelength == pytest.approx(1.06)
    assert bandwidth == pytest.approx(100 * 0.18 / 1.06)

    # A given wavelength sets the reference for the bandwidth
    wavelength, bandwidth = laue.pink_indexer_wavelength_and_bandwidth(
        _laue_beam(), wavelength=1.2
    )
    assert wavelength == pytest.approx(1.2)
    assert bandwidth == pytest.approx(100 * 0.18 / 1.2)


def test_set_pink_indexer_wavelength_defaults():
    experiments = ExperimentList([Experiment(beam=_laue_beam())])
    params = index_phil.fetch(sources=laue.index_defaults).extract()
    laue.set_pink_indexer_wavelength_defaults(params, experiments)
    assert params.indexing.pink_indexer.wavelength == pytest.approx(1.04)
    assert params.indexing.pink_indexer.percent_bandwidth == pytest.approx(
        17.31, abs=0.01
    )

    # Values set by the user are kept
    params = index_phil.fetch(sources=laue.index_defaults).extract()
    params.indexing.pink_indexer.wavelength = 1.0
    params.indexing.pink_indexer.percent_bandwidth = 5.0
    laue.set_pink_indexer_wavelength_defaults(params, experiments)
    assert params.indexing.pink_indexer.wavelength == 1.0
    assert params.indexing.pink_indexer.percent_bandwidth == 5.0


def test_index_laue_default_overrides():
    parser = ArgumentParser(phil=index_phil)
    params, _ = parser.parse_args(
        args=[
            # The standard default value, set explicitly, must win over the
            # Laue default
            "indexing.stills.rmsd_min_px=2",
            "indexing.pink_indexer.max_refls=60",
        ]
    )
    assert params.indexing.method == "fft3d"

    params = parser.apply_default_overrides(params, laue.index_defaults)
    assert params.indexing.stills.rmsd_min_px == 2
    assert params.indexing.pink_indexer.max_refls == 60
    # Laue defaults
    assert params.indexing.method == "pink_indexer"
    assert params.indexing.pink_indexer.wavelength is None
    assert params.indexing.pink_indexer.percent_bandwidth is None
    assert params.indexing.pink_indexer.target_lattices == 5
    assert params.indexing.pink_indexer.rotogram_grid_points == 360
    assert params.indexing.pink_indexer.voxel_grid_points == 250
    assert params.indexing.refinement_protocol.mode is None
    assert params.indexing.stills.ewald_proximal_volume_max == 0.03
    assert params.indexing.stills.refine_all_candidates is True
    assert params.refinement.parameterisation.crystal.fix == "cell"
    assert params.refinement.reflections.weighting_strategy.wavelength_weight == 0
    assert params.indexing.joint_indexing is False
    assert params.indexing.nproc == 1
    # Pipeline overrides of the laue.index defaults
    assert params.refinement.reflections.outlier.algorithm is None
    # laue.index defaults that the pipeline does not override
    assert params.refinement.reflections.reflections_per_degree is libtbx.Auto
    assert params.refinement.reflections.outlier.tukey.iqr_multiplier == 0
    assert params.refinement.reflections.outlier.minimum_number_of_reflections == 1
    assert params.refinement.parameterisation.auto_reduction.action == "fix"
    assert params.refinement.parameterisation.beam.fix == ["all"]
    assert params.refinement.parameterisation.detector.fix == "all"
    assert params.refinement.parameterisation.scan_varying is False

    # The modified parameters are reported relative to the standard defaults
    diff = parser.diff_phil.as_str()
    assert "pink_indexer" in diff
    assert "max_refls = 60" in diff


def test_find_spots_laue_default_overrides():
    parser = ArgumentParser(phil=find_spots_phil)
    params, _ = parser.parse_args(
        args=["output.shoeboxes=True", "spotfinder.filter.min_spot_size=3"]
    )
    params = parser.apply_default_overrides(params, laue.find_spots_defaults)
    assert params.spotfinder.force_2d is True
    assert params.spotfinder.filter.min_spot_size == 3
    # Set explicitly by the user, even though it is the standard default
    assert params.output.shoeboxes is True

    params, _ = parser.parse_args(args=[])
    params = parser.apply_default_overrides(params, laue.find_spots_defaults)
    assert params.spotfinder.force_2d is True
    assert params.output.shoeboxes is False
    assert params.spotfinder.filter.min_spot_size == 6
    # dials.find_spots-specific overrides are kept
    assert params.spotfinder.mp.nproc is libtbx.Auto


def test_refine_laue_default_overrides():
    parser = ArgumentParser(phil=refine_phil)
    params, _ = parser.parse_args(
        args=[
            "refinement.reflections.weighting_strategy.override=stills",
            "refinement.reflections.outlier.algorithm=tukey",
        ]
    )
    params = parser.apply_default_overrides(params, laue.refine_defaults)
    refinement = params.refinement
    assert refinement.reflections.weighting_strategy.override == "stills"
    assert refinement.reflections.outlier.algorithm == "tukey"
    assert refinement.reflections.outlier.minimum_number_of_reflections == 1

    params, _ = parser.parse_args(args=[])
    params = parser.apply_default_overrides(params, laue.refine_defaults)
    refinement = params.refinement
    assert refinement.reflections.weighting_strategy.override == "statistical"
    assert refinement.reflections.weighting_strategy.wavelength_weight == 100
    # The laue.refine defaults
    outlier = refinement.reflections.outlier
    assert outlier.nproc == 1
    assert outlier.minimum_number_of_reflections == 1
    assert outlier.algorithm == "mcd"
    assert outlier.separate_images is True
    parameterisation = refinement.parameterisation
    assert parameterisation.beam.fix == ["in_spindle_plane", "out_spindle_plane"]
    # Worked out from the crystal model by set_unit_cell_fix_list_defaults
    assert parameterisation.crystal.unit_cell.fix_list is libtbx.Auto
    assert parameterisation.detector.fix == "distance"
    assert parameterisation.auto_reduction.action == "fix"
    assert parameterisation.auto_reduction.min_nref_per_parameter == 1
    assert parameterisation.spherical_relp_model is True
    # dials.refine defaults that laue.refine does not change are kept
    assert parameterisation.scan_varying is libtbx.Auto
    # SparseLevMar in laue.refine, but DIALS has no sparse Laue target
    assert refinement.refinery.engine == "LevMar"
    assert params.output.experiments == "refined.expt"


def _crystal(space_group, cell):
    # Real space axes are the columns of the orthogonalisation matrix
    axes = matrix.sqr(uctbx.unit_cell(cell).orthogonalization_matrix()).transpose()
    return Crystal(
        axes[0:3],
        axes[3:6],
        axes[6:9],
        space_group=sgtbx.space_group_info(space_group).group(),
    )


@pytest.mark.parametrize(
    "space_group,cell,expected",
    [
        ("P 21 21 21", (34.297, 45.552, 99.035, 90, 90, 90), "g_param_0"),
        ("P 1 2 1", (34, 45, 99, 90, 105, 90), "g_param_0"),
        ("P 1", (34, 45, 99, 80, 95, 100), "g_param_0"),
        ("P 4", (45, 45, 99, 90, 90, 90), "g_param_0"),
        ("P 6", (45, 45, 99, 90, 90, 120), "g_param_1"),
        ("P 2 3", (45, 45, 45, 90, 90, 90), "g_param_0"),
    ],
)
def test_unit_cell_a_axis_parameter(space_group, cell, expected):
    assert laue.unit_cell_a_axis_parameter(_crystal(space_group, cell)) == expected


def test_set_unit_cell_fix_list_defaults():
    crystal = _crystal("P 6", (45, 45, 99, 90, 90, 120))
    experiments = ExperimentList([Experiment(beam=_laue_beam(), crystal=crystal)])
    parser = ArgumentParser(phil=refine_phil)

    params, _ = parser.parse_args(args=[])
    params = parser.apply_default_overrides(params, laue.refine_defaults)
    laue.set_unit_cell_fix_list_defaults(params, experiments)
    assert params.refinement.parameterisation.crystal.unit_cell.fix_list == [
        "g_param_1"
    ]

    # Set explicitly by the user
    for value, expected in (("g_param_0", ["g_param_0"]), ("None", None)):
        params, _ = parser.parse_args(
            args=[f"refinement.parameterisation.crystal.unit_cell.fix_list={value}"]
        )
        params = parser.apply_default_overrides(params, laue.refine_defaults)
        laue.set_unit_cell_fix_list_defaults(params, experiments)
        assert params.refinement.parameterisation.crystal.unit_cell.fix_list == expected
