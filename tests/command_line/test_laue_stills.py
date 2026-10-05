"""
Tests for importing pink-beam Laue data from NeXus (NXmx) files as stills, and
for the Laue defaults used by dials.find_spots for such data.
"""

from __future__ import annotations

import shutil
import subprocess

import h5py
import numpy as np
import pytest

from dxtbx.model import PolychromaticBeam
from dxtbx.model.experiment_list import ExperimentList

from dials.array_family import flex

pytest.importorskip("dxtbx.format.FormatNXmxLaue")

N_IMAGES = 3
IMAGE_SIZE = 100


@pytest.fixture
def nxmx_laue(tmp_path):
    """
    A small NXmx file with a pink-beam (polychromatic) incident wavelength and
    a few strong spots on each image.
    """
    filename = tmp_path / "laue.h5"
    with h5py.File(filename, "w") as f:
        entry = f.create_group("/entry")
        entry.attrs["NX_class"] = "NXentry"
        entry["definition"] = "NXmx"
        entry["start_time"] = "2021-09-10T06:54:37Z"
        entry["end_time"] = "2021-09-10T06:55:09Z"
        entry["end_time_estimated"] = "2021-09-10T06:55:09Z"

        source = entry.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source["name"] = "Synchrotron"

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        instrument["name"] = np.bytes_("LAUE BEAMLINE")

        beam = instrument.create_group("beam")
        beam.attrs["NX_class"] = "NXbeam"
        beam["incident_wavelength"] = np.array([0.97, 1.15])
        beam["incident_wavelength"].attrs["units"] = b"angstrom"
        beam["peak_wavelength"] = 1.04
        beam["peak_wavelength"].attrs["units"] = b"angstrom"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector["depends_on"] = "/entry/instrument/detector/transformations/det_z"
        detector["description"] = "Test detector"
        detector["sensor_material"] = "Silicon"
        detector["sensor_thickness"] = 0.00045
        detector["sensor_thickness"].attrs["units"] = b"m"
        detector["x_pixel_size"] = 1e-4
        detector["y_pixel_size"] = 1e-4
        detector["underload_value"] = -1
        detector["saturation_value"] = 1e6
        detector["count_time"] = 0.1
        detector["frame_time"] = 0.1
        detector["frame_time"].attrs["units"] = "s"
        detector["bit_depth_readout"] = np.array(32)
        detector["pixel_mask"] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

        transformations = detector.create_group("transformations")
        transformations.attrs["NX_class"] = "NXtransformations"
        det_z = transformations.create_dataset("det_z", data=np.array([100.0]))
        det_z.attrs["depends_on"] = b"."
        det_z.attrs["transformation_type"] = b"translation"
        det_z.attrs["units"] = b"mm"
        det_z.attrs["vector"] = np.array([0.0, 0.0, 1.0])

        module = detector.create_group("module")
        module.attrs["NX_class"] = "NXdetector_module"
        module["data_origin"] = np.array([0, 0])
        module["data_size"] = np.array([IMAGE_SIZE, IMAGE_SIZE])
        for name, vector in (
            ("fast_pixel_direction", [-1.0, 0.0, 0.0]),
            ("slow_pixel_direction", [0.0, -1.0, 0.0]),
        ):
            direction = module.create_dataset(name, data=1e-4)
            direction.attrs["transformation_type"] = "translation"
            direction.attrs["depends_on"] = (
                "/entry/instrument/detector/module/module_offset"
            )
            direction.attrs["vector"] = np.array(vector)
            direction.attrs["offset"] = np.array([0.0, 0.0, 0.0])
            direction.attrs["offset_units"] = b"m"
            direction.attrs["units"] = b"m"
        module_offset = module.create_dataset("module_offset", data=0)
        module_offset.attrs["transformation_type"] = "translation"
        module_offset.attrs["depends_on"] = detector["depends_on"]
        module_offset.attrs["vector"] = np.array([1.0, 0.0, 0.0])
        module_offset.attrs["offset"] = np.array([0.005, 0.005, 0.0])
        module_offset.attrs["offset_units"] = b"m"
        module_offset.attrs["units"] = b"m"

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample["name"] = "mysample"
        sample["depends_on"] = b"/entry/sample/transformations/omega"
        sample_transformations = sample.create_group("transformations")
        sample_transformations.attrs["NX_class"] = "NXtransformations"
        omega = sample_transformations.create_dataset(
            "omega", data=np.arange(N_IMAGES, dtype=float)
        )
        omega.attrs["depends_on"] = b"."
        omega.attrs["transformation_type"] = b"rotation"
        omega.attrs["units"] = b"deg"
        omega.attrs["vector"] = np.array([-1.0, 0.0, 0.0])
        sample_transformations["omega_increment_set"] = 1.0

        # Images of Poisson background with a few strong spots on each
        rng = np.random.default_rng(seed=42)
        images = rng.poisson(5, size=(N_IMAGES, IMAGE_SIZE, IMAGE_SIZE)).astype(
            np.int32
        )
        for image in images:
            for y, x in rng.integers(10, IMAGE_SIZE - 10, size=(5, 2)):
                image[y - 1 : y + 2, x - 1 : x + 2] += 500
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "data"
        data["data"] = images

    return filename


def _run(command, cwd):
    result = subprocess.run(
        [shutil.which(command[0]), *command[1:]],
        cwd=cwd,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr, result.stderr.decode()
    return result


def test_import_nexus_laue_as_stills(nxmx_laue, tmp_path):
    result = _run(["dials.import", nxmx_laue], tmp_path)
    assert b"importing" in result.stdout and b"as Laue stills" in result.stdout

    experiments = ExperimentList.from_file(tmp_path / "imported.expt")
    assert len(experiments) == N_IMAGES
    assert len(experiments.imagesets()) == N_IMAGES
    assert len(experiments.beams()) == 1
    for experiment in experiments:
        assert isinstance(experiment.beam, PolychromaticBeam)
        assert experiment.beam.get_wavelength_range() == pytest.approx((0.97, 1.15))
        assert experiment.beam.get_peak_wavelength() == pytest.approx(1.04)
        assert experiment.scan is None
        assert experiment.goniometer is None
    assert experiments.all_laue()


def test_import_nexus_laue_as_sequence(nxmx_laue, tmp_path):
    _run(["dials.import", nxmx_laue, "laue.import_as_stills=False"], tmp_path)

    experiments = ExperimentList.from_file(tmp_path / "imported.expt")
    assert len(experiments) == 1
    assert isinstance(experiments[0].beam, PolychromaticBeam)
    assert experiments[0].scan.get_num_images() == N_IMAGES
    assert experiments[0].goniometer is not None


def test_import_nexus_laue_with_geometry_override(nxmx_laue, tmp_path):
    _run(["dials.import", nxmx_laue, "distance=150"], tmp_path)

    experiments = ExperimentList.from_file(tmp_path / "imported.expt")
    assert len(experiments) == N_IMAGES
    for experiment in experiments:
        assert experiment.scan is None
        assert experiment.detector[0].get_distance() == pytest.approx(150)


def test_find_spots_laue_stills(nxmx_laue, tmp_path):
    _run(["dials.import", nxmx_laue], tmp_path)
    result = _run(["dials.find_spots", "imported.expt"], tmp_path)
    assert b"Laue stills found" in result.stdout

    reflections = flex.reflection_table.from_file(tmp_path / "strong.refl")
    assert len(reflections) > 0
    # output.shoeboxes=False by default for Laue stills
    assert "shoebox" not in reflections
    assert set(reflections["id"]) == set(range(N_IMAGES))
    # Spots start with the peak wavelength of the beam
    assert list(reflections["wavelength"]) == pytest.approx([1.04] * len(reflections))

    # Explicit user settings take precedence over the Laue defaults
    _run(
        [
            "dials.find_spots",
            "imported.expt",
            "output.shoeboxes=True",
            "output.reflections=strong_shoeboxes.refl",
        ],
        tmp_path,
    )
    reflections = flex.reflection_table.from_file(tmp_path / "strong_shoeboxes.refl")
    assert "shoebox" in reflections
