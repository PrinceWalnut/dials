"""
Joint optimisation of Miller indices, wavelengths and crystal orientations for
indexed Laue stills, as performed by laue.optimize_indexing from laue-dials.

For a polychromatic beam, the wavelength of each reflection is not known from
indexing. Each still is reindexed by assigning to each observed spot the
reciprocal lattice point, within the Ewald shell of the beam wavelength range,
whose direction is closest to that of the spot (as a linear sum assignment),
which gives the wavelength of the reflection. The crystal orientation is then
updated to superpose the predicted on the observed reciprocal lattice vectors,
outliers are rejected, and this is repeated for a number of macrocycles.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import MinCovDet

import libtbx
import libtbx.phil
from cctbx import crystal, miller
from dxtbx.model import ExperimentList
from scitbx import matrix

from dials.array_family import flex
from dials.util import tabulate

logger = logging.getLogger(__name__)

phil_scope = libtbx.phil.parse(
    """
assign_wavelengths = True
  .type = bool
  .help = "For Laue stills, after indexing, jointly optimise the Miller index"
          "assignments, the wavelength of each reflection and the crystal"
          "orientations, as laue.optimize_indexing from laue-dials does."
n_macrocycles = 5
  .type = int(value_min=0)
  .help = "The number of macrocycles of index optimisation. 0 performs a single"
          "assignment, without outlier rejection or crystal reorientation."
d_min = Auto
  .type = float(value_min=0)
  .help = "The resolution limit of the reciprocal lattice points that spots can"
          "be assigned to. If Auto, for each image, the best resolution that any"
          "of its spots could have, at the minimum wavelength of the spectrum."
wavelength_min = None
  .type = float(value_min=0)
  .help = "The minimum wavelength of the beam spectrum. If None, the lower end"
          "of the wavelength range of the beam."
wavelength_max = None
  .type = float(value_min=0)
  .help = "The maximum wavelength of the beam spectrum. If None, the upper end"
          "of the wavelength range of the beam."
filter_spectrum = True
  .type = bool
  .help = "Mark reflections assigned a wavelength outside the beam spectrum"
          "as unindexed."
outlier_nstd = 10.0
  .type = float(value_min=0)
  .help = "Reflections further than this many standard deviations (in Mahalanobis"
          "distance) from the observed and predicted reciprocal lattice vectors"
          "are rejected as outliers in each macrocycle."
  .expert_level = 1
""",
)


def _angle_between(u, v):
    """The angles, in radians, between each row of u and each row of v"""
    u = u / np.linalg.norm(u, axis=-1)[:, None]
    v = v / np.linalg.norm(v, axis=-1)[:, None]
    return np.arccos(np.clip(u @ v.T, -1.0, 1.0))


def _central_rays(hkl):
    """The shortest Miller index on the central ray of each Miller index"""
    gcd = np.gcd.reduce(hkl.astype(int), axis=-1)
    return hkl // gcd[:, None]


class LaueAssigner:
    """
    Assign Miller indices and wavelengths to the spots of a Laue still.

    A port of the LaueAssigner of laue-dials.
    """

    def __init__(
        self,
        unit_s0,
        s1,
        crystal_model,
        wavelength_min,
        wavelength_max,
        wavelength_peak,
        d_min,
        outlier_nstd=10.0,
    ):
        """
        Args:
            unit_s0: The unit vector along the incident beam.
            s1: An n x 3 array along the directions of the scattered beams.
            crystal_model: The crystal model, giving the unit cell, space group,
                and initial orientation.
            wavelength_min: The minimum wavelength of the beam spectrum.
            wavelength_max: The maximum wavelength of the beam spectrum.
            wavelength_peak: The wavelength used to choose between harmonics.
            d_min: The resolution limit of the reciprocal lattice points.
            outlier_nstd: The outlier rejection limit, in standard deviations.
        """
        self.s0 = np.asarray(unit_s0) / np.linalg.norm(unit_s0)
        self.U = np.array(crystal_model.get_U()).reshape(3, 3)
        self.B = np.array(crystal_model.get_B()).reshape(3, 3)
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.wavelength_peak = wavelength_peak
        self.outlier_nstd = outlier_nstd

        self._s1 = s1 / np.linalg.norm(s1, axis=-1)[:, None]
        self._qobs = self._s1 - self.s0
        self._qpred = np.zeros_like(self._s1)
        self._H = np.zeros(self._s1.shape, dtype=int)
        self._wavelength = np.zeros(len(self._s1))
        self._inliers = np.ones(len(self._s1), dtype=bool)

        self.Hall = self._reciprocal_grid(crystal_model, d_min)

    @staticmethod
    def _reciprocal_grid(crystal_model, d_min):
        """All Miller indices to d_min that are not systematically absent"""
        unit_cell = crystal_model.get_unit_cell()
        hmax, kmax, lmax = unit_cell.max_miller_indices(d_min)
        H = (
            np.mgrid[-hmax : hmax + 1, -kmax : kmax + 1, -lmax : lmax + 1]
            .reshape((3, -1))
            .T
        )
        H = H[np.any(H != 0, axis=1)]
        indices = flex.miller_index([tuple(int(i) for i in h) for h in H])
        ms = miller.set(
            crystal.symmetry(
                unit_cell=unit_cell, space_group=crystal_model.get_space_group()
            ),
            indices,
            anomalous_flag=True,
        )
        keep = (ms.d_spacings().data() >= d_min) & ~ms.sys_absent_flags().data()
        return H[np.array(keep, dtype=bool)]

    @property
    def qobs(self):
        return self._qobs[self._inliers]

    @property
    def qpred(self):
        return self._qpred[self._inliers]

    @property
    def wavelength(self):
        return self._wavelength[self._inliers]

    @property
    def inliers(self):
        return self._inliers.copy()

    @property
    def miller_indices(self):
        """The assigned Miller indices, (0, 0, 0) for unassigned spots"""
        return self._H.copy()

    @property
    def wavelengths(self):
        """The assigned wavelengths, 0 for unassigned spots"""
        return self._wavelength.copy()

    def reset_inliers(self):
        self._inliers = np.ones(len(self._inliers), dtype=bool)

    def _set_inlier_values(self, H, qpred, wavelength):
        self._H[self._inliers] = H
        self._H[~self._inliers] = 0
        self._qpred[self._inliers] = qpred
        self._wavelength[self._inliers] = wavelength
        self._wavelength[~self._inliers] = 0.0

    def assign(self):
        """Assign Miller indices and wavelengths to the inlying spots"""
        qall = self.Hall @ (self.U @ self.B).T
        feasible = (
            np.linalg.norm(qall + self.s0 / self.wavelength_min, axis=-1)
            < 1 / self.wavelength_min
        ) & (
            np.linalg.norm(qall + self.s0 / self.wavelength_max, axis=-1)
            > 1 / self.wavelength_max
        )
        Hall = self.Hall[feasible]
        qall = qall[feasible]

        # Of reflections on the same central ray (harmonics), keep the one with
        # the wavelength closest to the peak wavelength
        _, ray = np.unique(_central_rays(Hall), axis=0, return_inverse=True)
        ray = ray.ravel()
        harmonic_wavelength = -2.0 * (qall @ self.s0) / (qall * qall).sum(-1)
        order = np.lexsort((np.abs(harmonic_wavelength - self.wavelength_peak), ray))
        first = np.ones(len(order), dtype=bool)
        first[1:] = ray[order][1:] != ray[order][:-1]
        keep = order[first]
        Hall = Hall[keep]
        qall = qall[keep]

        # Assign spots to reciprocal lattice points by direction
        qobs = self.qobs
        rows, cols = linear_sum_assignment(_angle_between(qobs, qall))
        if len(rows) < len(qobs):
            # More spots than feasible reflections: leave the rest unassigned
            assigned = np.zeros(len(qobs), dtype=bool)
            assigned[rows] = True
            self._inliers[self._inliers] = assigned
        H = Hall[cols]
        qpred = qall[cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            wavelength = np.linalg.norm(qobs[rows], axis=-1) / np.linalg.norm(
                qpred, axis=-1
            )
        self._set_inlier_values(H, qpred, wavelength)

    def reject_outliers(self):
        """Reject outliers by Mahalanobis distance, using a robust covariance"""
        X = np.concatenate((self.qobs, self.qpred * self.wavelength[:, None]), axis=-1)
        if len(X) <= X.shape[1]:
            return
        dist = MinCovDet(random_state=0).fit(X).dist_
        self._inliers[self._inliers] = dist <= self.outlier_nstd**2

    def update_rotation(self):
        """Rotate the crystal to superpose predicted on observed vectors"""
        misset, _ = orthogonal_procrustes(
            self.qobs, self.qpred * self.wavelength[:, None]
        )
        self.U = misset @ self.U

    def optimise(self, n_macrocycles):
        self.assign()
        for _ in range(n_macrocycles):
            self.reset_inliers()
            self.update_rotation()
            self.assign()
            self.reject_outliers()
            self.update_rotation()
            self.assign()


def auto_d_min(unit_s0, s1, wavelength_min):
    """
    The best resolution that any of the spots could have: that of the spot at
    the highest scattering angle, at the minimum wavelength of the spectrum.

    Args:
        unit_s0: The unit vector along the incident beam.
        s1: An n x 3 array along the directions of the scattered beams.
        wavelength_min: The minimum wavelength of the beam spectrum.
    """
    s1 = s1 / np.linalg.norm(s1, axis=-1)[:, None]
    cos_two_theta = np.clip(s1 @ np.asarray(unit_s0), -1.0, 1.0)
    sin_theta = np.sqrt((1.0 - cos_two_theta.min()) / 2.0)
    return wavelength_min / (2.0 * sin_theta)


def _imageset_ids(experiments, reflections):
    """The imageset_id of the reflections of each experiment"""
    imageset_ids = []
    for i in range(len(experiments)):
        sel = reflections["id"] == i
        ids = set(reflections["imageset_id"].select(sel))
        assert len(ids) <= 1, "An experiment has reflections on several imagesets"
        imageset_ids.append(ids.pop() if ids else None)
    return imageset_ids


def optimise_laue_indexing(
    experiments: ExperimentList,
    reflections: flex.reflection_table,
    imported_experiments: ExperimentList,
    params,
):
    """
    Optimise the Miller indices, wavelengths and crystal orientations of indexed
    Laue stills.

    Each indexed experiment is reoptimised from its indexed reflections, together
    with the unindexed reflections on its image if it is the only experiment on
    that image. Reflections that are not assigned are left unindexed (id -1,
    Miller index (0, 0, 0)), with the peak wavelength of the beam.

    The wavelength and s0 columns of the reflections are set to the assigned
    wavelength of each reflection, and the s1 and rlp columns are recalculated
    from them. The crystal orientations of the experiments are updated in place.

    Args:
        experiments: The indexed experiments.
        reflections: The indexed (and unindexed) reflections, with imageset_id
            indexing the imported experiments.
        imported_experiments: The experiments indexing was given, one for each
            imageset.
        params: The laue scope of the dials.index parameters.

    Returns:
        The optimised reflections.
    """
    reflections = reflections.copy()
    imageset_ids = _imageset_ids(experiments, reflections)
    n_expts_on_imageset = {}
    for imageset_id in imageset_ids:
        n_expts_on_imageset[imageset_id] = n_expts_on_imageset.get(imageset_id, 0) + 1

    unindexed = reflections["id"] == -1
    wavelength = reflections["wavelength"]
    s0 = reflections["s0"]
    indexed_flag = reflections.flags.indexed

    logger.info("\nOptimising Laue indexing (as laue.optimize_indexing)")
    rows = []
    for i, (expt, imageset_id) in enumerate(zip(experiments, imageset_ids)):
        sel = reflections["id"] == i
        if imageset_id is not None and n_expts_on_imageset[imageset_id] == 1:
            sel |= unindexed & (reflections["imageset_id"] == imageset_id)
        isel = sel.iselection()
        if len(isel) == 0:
            continue
        n_before = (reflections["id"].select(isel) == i).count(True)

        beam = expt.beam
        beam_min, beam_max = beam.get_wavelength_range()
        wavelength_min = params.wavelength_min or beam_min
        wavelength_max = params.wavelength_max or beam_max
        wavelength_peak = beam.get_peak_wavelength()
        unit_s0 = matrix.col(beam.get_unit_s0())

        # Directions of the scattered beams from the observed centroids
        refls = reflections.select(isel)
        s1 = flex.vec3_double(len(refls))
        x, y, _ = refls["xyzobs.mm.value"].parts()
        for i_panel, panel in enumerate(expt.detector):
            psel = refls["panel"] == i_panel
            s1.set_selected(
                psel,
                panel.get_lab_coord(flex.vec2_double(x.select(psel), y.select(psel))),
            )

        s1 = s1.as_numpy_array()
        if params.d_min is libtbx.Auto:
            d_min = auto_d_min(unit_s0, s1, wavelength_min)
        else:
            d_min = params.d_min

        assigner = LaueAssigner(
            unit_s0,
            s1,
            expt.crystal,
            wavelength_min,
            wavelength_max,
            wavelength_peak,
            d_min,
            params.outlier_nstd,
        )
        U_before = matrix.sqr(expt.crystal.get_U())
        assigner.optimise(params.n_macrocycles)
        U_after = matrix.sqr(tuple(assigner.U.ravel()))
        expt.crystal.set_U(U_after)

        assigned_wavelength = assigner.wavelengths
        assigned = assigned_wavelength > 0
        if params.filter_spectrum:
            assigned &= (assigned_wavelength >= wavelength_min) & (
                assigned_wavelength <= wavelength_max
            )
        hkl = assigner.miller_indices
        hkl[~assigned] = 0
        assigned_wavelength[~assigned] = wavelength_peak

        assigned_flex = flex.bool(assigned.tolist())
        new_id = flex.int(len(isel), -1)
        new_id.set_selected(assigned_flex, i)
        reflections["id"].set_selected(isel, new_id)
        reflections["miller_index"].set_selected(
            isel, flex.miller_index([tuple(int(j) for j in h) for h in hkl])
        )
        wavelength.set_selected(isel, flex.double(assigned_wavelength.tolist()))
        s0.set_selected(
            isel,
            flex.vec3_double(
                [tuple(unit_s0 / w) for w in assigned_wavelength.tolist()]
            ),
        )
        for mask, set_or_unset in (
            (assigned_flex, reflections.set_flags),
            (~assigned_flex, reflections.unset_flags),
        ):
            full_mask = flex.bool(len(reflections), False)
            full_mask.set_selected(isel.select(mask), True)
            set_or_unset(full_mask, indexed_flag)

        rotation = (
            U_after * U_before.transpose()
        ).r3_rotation_matrix_as_unit_quaternion()
        angle = math.degrees(2 * math.acos(min(1.0, abs(rotation[0]))))
        w = assigned_wavelength[assigned]
        rows.append(
            [
                str(i),
                f"{d_min:.2f}",
                str(n_before),
                str(int(assigned.sum())),
                f"{w.min():.3f}" if len(w) else "-",
                f"{w.max():.3f}" if len(w) else "-",
                f"{angle:.3f}",
            ]
        )

    # Recalculate s1 and rlp from the assigned wavelengths
    reflections.map_centroids_to_reciprocal_space(imported_experiments)

    logger.info(
        tabulate(
            rows,
            [
                "Experiment",
                "d_min\n(Å)",
                "Indexed\nbefore",
                "Indexed\nafter",
                "Wavelength\nmin (Å)",
                "Wavelength\nmax (Å)",
                "Rotation\n(°)",
            ],
        )
    )
    return reflections
