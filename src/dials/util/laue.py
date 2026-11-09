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

import libtbx
import libtbx.phil
from dxtbx.model import ExperimentList, PolychromaticBeam

logger = logging.getLogger(__name__)

# dials.find_spots defaults for Laue stills (as in laue.find_spots), plus the
# min_spot_size that min_spot_size=Auto gives for the CCD (Rayonix) data used in
# the laue-dials serial pink-beam pipeline. Auto would otherwise give 3 for
# NeXus data, as dxtbx reads every NeXus detector as a pixel-array detector.
find_spots_defaults = libtbx.phil.parse(
    """
spotfinder {
  force_2d = True
  filter {
    min_spot_size = 6
  }
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
    # As in the pipeline. Candidates are refined with Laue refinement, without
    # the Nave mosaicity model of monochromatic stills (see the refinement
    # settings below)
    refine_all_candidates = True
  }
  joint_indexing = False
}
refinement {
  parameterisation {
    auto_reduction {
      action = fix
    }
    scan_varying = False
    # Not from the pipeline: for refining the candidates of Laue stills, which
    # is by spot position only (below), fix the unit cell, as an overall scale
    # of the cell cannot then be determined. The known cell is kept for the
    # wavelength assignment after indexing (as laue.optimize_indexing does
    # with geometry.unit_cell in the pipeline).
    crystal {
      fix = cell
    }
  }
  reflections {
    outlier {
      algorithm = None
    }
    # Not from the pipeline: refine the candidates by spot position only, as
    # every spot still has the peak wavelength of the beam when indexing
    weighting_strategy {
      wavelength_weight = 0
    }
  }
}
"""
)

# The dials.index Laue defaults, in the order they are applied
index_defaults = [laue_index_defaults, serial_pipeline_index_overrides]

# dials.refine defaults for Laue stills: those of the laue.refine program from
# laue-dials, which the laue-dials serial pink-beam pipeline
# (examples/serial_pipeline.sh) uses unchanged, except for the weighting
# strategy. laue.refine uses stills weighting, as it refines a monochromatic
# beam for each reflection; statistical weighting is used here, which DIALS
# refinement uses as Laue statistical weighting for a polychromatic beam, with
# a wavelength_weight of 100 rather than the DIALS default of 1e4. At 1e4 the
# wavelength term, against the fixed wavelengths assigned in indexing,
# dominates the target: on the DHFR serial data, the median spot position RMSD
# was about 1.07 px at 1e4 and 0.53 px at 100 (and at any weight down to 0.01),
# with the wavelength RMSD at 0.0014 and 0.0030 A.
# Not reproduced are the output filenames, the laue.refine nproc setting (the
# number of images refined in parallel) and refinery.engine=SparseLevMar.
#
# TODO: DIALS has no sparse Laue refinement target (nor sparse Laue prediction
# parameterisation), so SparseLevMar fails for Laue data and the default LevMar
# engine is kept for now. Add a sparse Laue target, then use SparseLevMar as
# laue.refine does.
refine_defaults = libtbx.phil.parse(
    """
refinement {
  reflections {
    weighting_strategy {
      override = statistical
      wavelength_weight = 100
    }
    outlier {
      nproc = 1
      minimum_number_of_reflections = 1
      algorithm = mcd
      separate_images = True
    }
  }
  parameterisation {
    beam {
      fix = *in_spindle_plane *out_spindle_plane
    }
    crystal {
      unit_cell {
        # laue.refine sets fix_list = real_space_a to fix the a-axis, but no
        # unit cell parameter name (g_param_0, g_param_1, ...) contains it, so
        # it fixes nothing. Auto is worked out from the crystal model by
        # set_unit_cell_fix_list_defaults, as the parameter for the a-axis
        fix_list = Auto
      }
    }
    detector {
      fix = distance
    }
    auto_reduction {
      action = fix
      min_nref_per_parameter = 1
    }
    spherical_relp_model = True
  }
}
"""
)


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


def unit_cell_a_axis_parameter(crystal) -> str:
    """
    The name of the refinement parameter of the crystal unit cell for its a-axis.

    The unit cell is refined with the independent elements of the reciprocal
    metrical matrix G* allowed by the space group, as parameters g_param_0,
    g_param_1, ... The parameter for the a-axis is the one on which G*_11
    (a*.a*) depends. Which this is depends on the lattice (for example,
    g_param_0 for an orthorhombic and g_param_1 for a hexagonal cell). Where
    the space group makes a and b (or a, b and c) equivalent, the parameter is
    shared by those axes.
    """
    from scitbx import matrix

    from dials.algorithms.refinement.parameterisation.crystal_parameters import (
        CrystalUnitCellParameterisation,
    )

    parameterisation = CrystalUnitCellParameterisation(crystal)
    B = matrix.sqr(crystal.get_B())
    # d(G*_11)/dp for G* = B^T B
    dG11_dp = [
        abs((dB.transpose() * B + B.transpose() * dB)[0])
        for dB in parameterisation.get_ds_dp()
    ]
    largest = max(dG11_dp)
    a_axis_parameters = [
        name
        for name, d in zip(parameterisation.get_param_names(), dG11_dp)
        if d > 1e-6 * largest
    ]
    assert len(a_axis_parameters) == 1, a_axis_parameters
    return a_axis_parameters[0]


def set_unit_cell_fix_list_defaults(params, experiments: ExperimentList):
    """
    Fill in a refinement unit cell fix_list of Auto to fix the a-axis
    parameter of the crystal unit cell, worked out from the crystal models.
    """
    unit_cell = params.refinement.parameterisation.crystal.unit_cell
    if unit_cell.fix_list is not libtbx.Auto:
        return
    crystals = experiments.crystals()
    if not crystals:
        unit_cell.fix_list = None
        return
    name = unit_cell_a_axis_parameter(crystals[0])
    if any(unit_cell_a_axis_parameter(crystal) != name for crystal in crystals[1:]):
        logger.warning(
            "Crystals have differing unit cell parameterisations: using the first "
            "crystal to choose the unit cell parameter to fix"
        )
    unit_cell.fix_list = [name]
    logger.info(
        "Setting refinement.parameterisation.crystal.unit_cell.fix_list=%s, "
        "the unit cell parameter for the a-axis",
        name,
    )
