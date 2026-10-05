"""
Helpers for recognising pink-beam Laue stills data and setting up default
processing parameters for it.

Pink-beam Laue data are described by a dxtbx PolychromaticBeam, which carries a
wavelength range and (optionally) the wavelength of maximum intensity in the
incident spectrum. dials.import converts such data from NeXus/NXmx files into
stills, and dials.find_spots and dials.index then switch to defaults suitable
for Laue stills, matching those used by the laue.find_spots and laue.index
programs from laue-dials.
"""

from __future__ import annotations

import logging

import libtbx.phil
from dxtbx.model import ExperimentList, PolychromaticBeam

logger = logging.getLogger(__name__)

# dials.find_spots defaults for Laue stills (as in laue.find_spots)
find_spots_defaults = libtbx.phil.parse(
    """
spotfinder {
  force_2d = True
}
output {
  shoeboxes = False
}
"""
)

# dials.index defaults for Laue stills come in two layers, applied in order:
#
# 1. The defaults of the laue.index program from laue-dials. laue.index wraps
#    the dials.index phil scope without the dials.index reflections_per_degree
#    override, and fixes the beam and detector in refinement. (Its
#    "refinement_procotol" setting is misspelt there, so has no effect, and is
#    not reproduced here.)
# 2. The pink_indexer settings from the laue-dials serial pink-beam pipeline
#    (serial_pipeline.sh), which take precedence over the laue.index defaults.
#
# Dataset-specific pipeline settings are not defaults: the unit cell and space
# group must be given by the user, and the pink_indexer wavelength and
# percent_bandwidth are left as None here so that they are worked out from the
# beam model by set_pink_indexer_wavelength_defaults (for the DHFR data in the
# pipeline this gives the pipeline values, 1.04 A and 17.31%).
laue_index_defaults = libtbx.phil.parse(
    """
refinement {
  parameterisation {
    beam {
      fix = all
    }
    detector {
      fix = all
    }
    goniometer {
      fix = all
    }
  }
  reflections {
    reflections_per_degree = Auto
    outlier {
      algorithm = tukey
      minimum_number_of_reflections = 1
      tukey {
        iqr_multiplier = 0.
      }
    }
  }
}
"""
)

serial_pipeline_index_overrides = libtbx.phil.parse(
    """
indexing {
  nproc = 1
  method = pink_indexer
  pink_indexer {
    wavelength = None
    percent_bandwidth = None
    max_refls = 50
    # min_lattices = 5 in the pipeline: min_lattices is deprecated in favour
    # of target_lattices, which it sets
    target_lattices = 5
    rotogram_grid_points = 360
    voxel_grid_points = 250
  }
  refinement_protocol {
    mode = None
  }
  stills {
    ewald_proximal_volume_max = 0.03
    rmsd_min_px = 20
    # True in the pipeline, but candidate refinement fails for every candidate
    # with Laue data, as the stills outlier rejection it relies on needs
    # delpsical.rad, which Laue refinement does not compute
    refine_all_candidates = False
  }
  joint_indexing = False
}
refinement {
  parameterisation {
    auto_reduction {
      action = fix
    }
    scan_varying = False
  }
  reflections {
    outlier {
      algorithm = None
    }
  }
}
"""
)

# The dials.index Laue defaults, in the order they are applied
index_defaults = [laue_index_defaults, serial_pipeline_index_overrides]


def is_laue_beam(beam) -> bool:
    """True if the beam is a polychromatic (Laue) beam."""
    return isinstance(beam, PolychromaticBeam)


def is_laue_stills(experiments: ExperimentList) -> bool:
    """
    True if the experiments are all Laue stills.

    That is, every experiment has a polychromatic beam, is not time-of-flight
    data, and has no scan (as for data imported as stills).
    """
    return (
        len(experiments) > 0
        and experiments.all_laue()
        and all(expt.scan is None for expt in experiments)
    )


def pink_indexer_wavelength_and_bandwidth(
    beam, wavelength: float | None = None
) -> tuple[float, float]:
    """
    The pink_indexer wavelength and percent bandwidth for a polychromatic beam.

    The wavelength is the peak wavelength of the incident spectrum (the
    wavelength range midpoint if no peak wavelength is recorded), unless one is
    given. The percent bandwidth is the full wavelength range of the beam as a
    percentage of that wavelength.
    """
    if wavelength is None:
        wavelength = beam.get_peak_wavelength()
    if wavelength <= 0:
        raise ValueError(
            "Cannot determine a pink_indexer wavelength from a polychromatic "
            "beam with no wavelength range: set indexing.pink_indexer.wavelength"
        )
    wavelength_min, wavelength_max = beam.get_wavelength_range()
    percent_bandwidth = 100.0 * (wavelength_max - wavelength_min) / wavelength
    return wavelength, percent_bandwidth


def set_pink_indexer_wavelength_defaults(params, experiments: ExperimentList):
    """
    Fill in the pink_indexer wavelength and percent bandwidth from the beam
    model, for whichever of the two are unset (None).
    """
    pink_params = params.indexing.pink_indexer
    if pink_params.wavelength is not None and pink_params.percent_bandwidth is not None:
        return
    beams = experiments.beams()
    wavelength, percent_bandwidth = pink_indexer_wavelength_and_bandwidth(
        beams[0], pink_params.wavelength
    )
    if any(
        pink_indexer_wavelength_and_bandwidth(beam, pink_params.wavelength)
        != (wavelength, percent_bandwidth)
        for beam in beams[1:]
    ):
        logger.warning(
            "Experiments have differing polychromatic beams: using the first "
            "beam to set the pink_indexer wavelength and bandwidth"
        )
    if pink_params.wavelength is None:
        pink_params.wavelength = wavelength
        logger.info(
            "Setting indexing.pink_indexer.wavelength=%.4f from the beam peak wavelength",
            wavelength,
        )
    if pink_params.percent_bandwidth is None:
        pink_params.percent_bandwidth = percent_bandwidth
        logger.info(
            "Setting indexing.pink_indexer.percent_bandwidth=%.2f from the beam "
            "wavelength range",
            percent_bandwidth,
        )
