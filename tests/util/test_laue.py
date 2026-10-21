from __future__ import annotations

import pytest

import libtbx
from dxtbx.model import BeamFactory, Experiment, ExperimentList, Scan

from dials.command_line.find_spots import working_phil as find_spots_phil
from dials.command_line.index import working_phil as index_phil
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
    assert params.indexing.stills.refine_all_candidates is False
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
