"""Contains classes used to provide weighting schemes as strategies for
ReflectionManagers."""

from __future__ import annotations

import math

import numpy as np

from dials.algorithms.refinement import DialsRefineConfigError
from dials.array_family import flex


class StatisticalWeightingStrategy:
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for refinement"""

    @staticmethod
    def calculate_weights(reflections):
        """set 'statistical weights', that is w(x) = 1/var(x)"""

        weights = (reflections["xyzobs.mm.variance"]).deep_copy()
        parts = weights.parts()
        for w in parts:
            sel = w > 0.0
            w.set_selected(sel, 1.0 / w.select(sel))
        reflections["xyzobs.mm.weights"] = flex.vec3_double(*parts)
        indexed = reflections.select(reflections.get_flags(reflections.flags.indexed))
        if any(indexed["xyzobs.mm.weights"].norms() == 0.0):
            raise DialsRefineConfigError(
                "Cannot set statistical weights as some indexed reflections have observed variances equal to zero"
            )

        return reflections


class StillsWeightingStrategy(StatisticalWeightingStrategy):
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for refinement. This version uses statistical weights
    for X and Y and a fixed constant for the delta Psi part, defaulting to 1000000"""

    def __init__(self, delpsi_constant=1000000):
        self._delpsi_constant = delpsi_constant

    def calculate_weights(self, reflections):
        """Include weights for DeltaPsi"""

        # call parent class method to set X and Y weights
        reflections = super().calculate_weights(reflections)

        reflections["delpsical.weights"] = flex.double(
            len(reflections), self._delpsi_constant
        )

        return reflections


class ExternalDelPsiWeightingStrategy(StatisticalWeightingStrategy):
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for stills refinement. This version uses statistical
    weights for X and Y and assume that the Delta Psi part is already provided in
    the reflection table"""

    def calculate_weights(self, reflections):
        """Statistical weights for X, Y. Weights for DeltaPsi must be already
        provided in the reflection table"""

        # call parent class method to set X and Y weights
        reflections = super().calculate_weights(reflections)

        if "delpsical.weights" not in reflections:
            raise DialsRefineConfigError(
                'The key "delpsical.weights" is expected within the input reflections'
            )

        return reflections


class ConstantWeightingStrategy:
    def __init__(self, wx, wy, wz, stills=False):
        self._wx = wx
        self._wy = wy
        self._wz = wz
        self._stills = stills

    def calculate_weights(self, reflections):
        """Set weights to constant terms. If stills, the z weights are
        the 'delpsical.weights' attribute of the reflection table. Otherwise, use
        the usual 'xyzobs.mm.weights'"""

        wx = flex.double(len(reflections), self._wx)
        wy = flex.double(len(reflections), self._wy)
        wz = flex.double(len(reflections), self._wz)
        if self._stills:
            null = flex.double(len(reflections), 0)
            reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, null)
            reflections["delpsical.weights"] = wz
        else:
            reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, wz)

        return reflections


class LaueStatisticalWeightingStrategy(StatisticalWeightingStrategy):
    """
    Variance in z estimated from sqrt(x^2+y^2)
    """

    def __init__(
        self,
        wavelength_weight: float = 1e7,
    ):
        self._wavelength_weight = wavelength_weight

    def calculate_weights(self, reflections):
        reflections = super().calculate_weights(reflections)

        wx, wy, _ = reflections["xyzobs.mm.weights"].parts()
        wz = flex.sqrt(wx * wx + wy * wy) * self._wavelength_weight
        reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, wz)

        return reflections


class LaueMixedWeightingStrategy(StatisticalWeightingStrategy):
    """
    Use statistical weighting for x and y, and constant weighting for z
    """

    def __init__(
        self,
        wavelength_weight: float = 1e7,
    ):
        self._wavelength_weight = wavelength_weight

    def calculate_weights(self, reflections):
        reflections = super().calculate_weights(reflections)

        wx, wy, wz = reflections["xyzobs.mm.weights"].parts()
        wz = wz * 0 + self._wavelength_weight
        reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, wz)

        return reflections


def _laue_spectrum_centre_and_sigma(beam):
    """
    The centre and Gaussian standard deviation of the spectrum of a
    polychromatic beam.

    The centre is the peak wavelength of the beam if it records one, otherwise
    the midpoint of its wavelength range. The wavelength range is taken as the
    full width at half maximum of a Gaussian spectrum, so that
    sigma = (wavelength_max - wavelength_min) / (2 sqrt(2 ln 2)).
    """
    wavelength_min, wavelength_max = beam.get_wavelength_range()
    centre = None
    if hasattr(beam, "get_peak_wavelength"):
        centre = beam.get_peak_wavelength()
    if not centre or centre <= 0:
        centre = 0.5 * (wavelength_min + wavelength_max)
    sigma = (wavelength_max - wavelength_min) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return centre, sigma


def _robust_centre_and_sigma(values):
    """The median and the normalised median absolute deviation of values"""
    values = np.asarray(values, dtype=float)
    centre = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - centre)))
    return centre, sigma


def laue_assigned_wavelength_variances(
    experiments, reflections, cell_relative_sigma=0.0, use_B_covariance=False
):
    """
    The variance of the wavelength assigned to each reflection of a Laue
    experiment, propagated from the variances of its observed centroid.

    The LaueAssigner assigns a wavelength by Bragg's law, from the scattering
    angle of the observed spot and the d-spacing of the assigned Miller index:
    wavelength_obs = |s1/|s1| - s0/|s0|| / |UB h| = 2 d sin(theta_obs). With d
    fixed, d(wavelength)/d(theta) = 2 d cos(theta) = wavelength / tan(theta), so

        var(wavelength) = (wavelength / tan(theta))^2 var(theta),

    where var(theta) = var(2 theta) / 4 is propagated from var(X) and var(Y)
    through the detector geometry (the covariance of X and Y is not known and
    is taken as zero).

    The wavelength is assigned with the d-spacings of the crystal model at
    indexing, whose error enters as var(wavelength) += wavelength^2 var(d)/d^2.
    This is added for an isotropic relative error of the d-spacings,
    cell_relative_sigma, and, if use_B_covariance and the crystal model has a
    B matrix covariance, for that covariance, propagated to |B h| = 1/d with
    d|Bh|/dB_ij = (B h)_i h_j / |B h|. Both treat the cell error of each
    reflection as independent, although it is common to all the reflections
    of a crystal.
    """
    n = len(reflections)
    variances = np.zeros(n)
    x, y, _ = reflections["xyzobs.mm.value"].parts()
    var_x, var_y, _ = reflections["xyzobs.mm.variance"].parts()
    x = x.as_numpy_array()
    y = y.as_numpy_array()
    var_x = var_x.as_numpy_array()
    var_y = var_y.as_numpy_array()
    wavelength = reflections["wavelength"].as_numpy_array()
    expt_ids = reflections["id"].as_numpy_array()
    panels = reflections["panel"].as_numpy_array()

    for i_expt, expt in enumerate(experiments):
        unit_s0 = np.array(expt.beam.get_unit_s0())
        in_expt = expt_ids == i_expt
        rel_var_d = np.zeros(int(in_expt.sum()))
        if cell_relative_sigma:
            rel_var_d += cell_relative_sigma**2
        cov = expt.crystal.get_B_covariance() if use_B_covariance else ()
        if len(cov) == 81:
            cov = np.array(cov).reshape(9, 9)
            B = np.array(expt.crystal.get_B()).reshape(3, 3)
            hkl = np.array(reflections["miller_index"].select(flex.bool(in_expt)))
            q = hkl @ B.T
            q_len_sq = (q * q).sum(axis=1)
            # Jacobian of |q| with respect to the elements of B (row major)
            J = (q[:, :, None] * hkl[:, None, :]).reshape(-1, 9) / np.sqrt(q_len_sq)[
                :, None
            ]
            rel_var_d += np.einsum("ni,ij,nj->n", J, cov, J) / q_len_sq
        variances[in_expt] += wavelength[in_expt] ** 2 * rel_var_d
        for i_panel, panel in enumerate(expt.detector):
            sel = in_expt & (panels == i_panel)
            if not sel.any():
                continue
            # Lab coordinates r = D (x, y, 1), with the fast and slow axes as
            # the first two columns of the d matrix
            D = np.array(panel.get_d_matrix()).reshape(3, 3)
            fast, slow = D[:, 0], D[:, 1]
            r = np.array(
                panel.get_lab_coord(
                    flex.vec2_double(flex.double(x[sel]), flex.double(y[sel]))
                )
            )
            r_len = np.linalg.norm(r, axis=1)
            u = r / r_len[:, None]
            cos_2theta = np.clip(u @ unit_s0, -1.0, 1.0)
            sin_2theta = np.sqrt(1.0 - cos_2theta**2)

            # d(2 theta)/dx = -d(cos 2 theta)/dx / sin(2 theta), where
            # d(r/|r|)/dx = (fast - u (u.fast)) / |r|
            with np.errstate(divide="ignore", invalid="ignore"):
                d2theta_dx = (
                    -(
                        ((fast[None, :] - u * (u @ fast)[:, None]) / r_len[:, None])
                        @ unit_s0
                    )
                    / sin_2theta
                )
                d2theta_dy = (
                    -(
                        ((slow[None, :] - u * (u @ slow)[:, None]) / r_len[:, None])
                        @ unit_s0
                    )
                    / sin_2theta
                )
            var_2theta = d2theta_dx**2 * var_x[sel] + d2theta_dy**2 * var_y[sel]
            theta = 0.5 * np.arccos(cos_2theta)
            with np.errstate(divide="ignore", invalid="ignore"):
                dlambda_dtheta = wavelength[sel] / np.tan(theta)
            variances[sel] += dlambda_dtheta**2 * var_2theta / 4.0

    # Spots along the beam have no defined wavelength: give them zero weight
    variances[~np.isfinite(variances)] = np.inf
    return flex.double(variances)


def _laue_q_lengths(experiments, reflections):
    """|A h| = 1/d for each reflection, from the static crystal models"""
    q_len = flex.double(len(reflections))
    h = reflections["miller_index"].as_vec3_double()
    ids = reflections["id"]
    for i_expt, expt in enumerate(experiments):
        sel = ids == i_expt
        n = sel.count(True)
        if n == 0:
            continue
        A = flex.mat3_double(n, expt.crystal.get_A())
        q_len.set_selected(sel, (A * h.select(sel)).norms())
    return q_len


def _laue_two_sin_theta_obs(experiments, reflections):
    """2 sin(theta) from the observed scattering directions ("s1")"""
    s1 = reflections["s1"]
    unit_s1 = s1 / s1.norms()
    unit_s0 = flex.vec3_double(len(reflections))
    ids = reflections["id"]
    for i_expt, expt in enumerate(experiments):
        sel = ids == i_expt
        unit_s0.set_selected(sel, expt.beam.get_unit_s0())
    return (unit_s1 - unit_s0).norms()


def laue_wavelength_obs_from_model(experiments, reflections):
    """
    The observed wavelengths for the current crystal models, for a correlated
    cell error: wavelength_obs = factor / |A h| = factor d, with the factor
    2 sin(theta_obs) ("wavelength_obs_factor") from the observed scattering
    directions. This is the wavelength the LaueAssigner would assign with the
    current cell. Uses the static crystal models.
    """
    return reflections["wavelength_obs_factor"] / _laue_q_lengths(
        experiments, reflections
    )


class LaueWavelengthVarianceWeightingStrategy(StatisticalWeightingStrategy):
    """
    Statistical weights for X and Y, and a weight for the wavelength term of
    the Laue target, w = 1 / var(wavelength), from a model of the variance of
    the observed wavelength, rather than from a fixed wavelength_weight.

    Two models are available:

    assignment: the observed wavelength is the wavelength assigned to each
        reflection in indexing (the "wavelength" column), and its variance is
        propagated from the variances of the observed centroid
        (laue_assigned_wavelength_variances).

    spectrum: a maximum-likelihood target for a Gaussian spectrum. The
        probability of observing a reflection at the wavelength
        wavelength_cal at which it is predicted to diffract is taken as
        proportional to the beam spectrum, N(wavelength_cal; centre, sigma^2).
        The negative log likelihood adds 1/(2 sigma^2) (wavelength_cal -
        centre)^2 (and a constant, as sigma is not refined) to the positional
        terms, i.e. the observed wavelength is the centre of the spectrum and
        w = 1 / sigma^2. The centre and sigma are taken for each beam either
        from the beam model (the peak wavelength, or the range midpoint, and the
        wavelength range as the FWHM) or from the distribution of the
        wavelengths assigned to its reflections (the median and the normalised
        median absolute deviation), unless given.

    The observed wavelength for the target is written to the "wavelength_obs"
    column, which the Laue target and reflection manager use in place of the
    "wavelength" column when it is present.
    """

    def __init__(
        self,
        experiments,
        model="spectrum",
        spectrum_source="beam",
        spectrum_centre=None,
        spectrum_sigma=None,
        min_sigma=1e-6,
        cell_relative_sigma=0.0,
        use_B_covariance=True,
        cell_error="independent",
    ):
        if model not in ("assignment", "spectrum"):
            raise ValueError(f"Unknown wavelength variance model {model}")
        if spectrum_source not in ("beam", "reflections"):
            raise ValueError(f"Unknown spectrum source {spectrum_source}")
        self._experiments = experiments
        self._model = model
        self._spectrum_source = spectrum_source
        self._spectrum_centre = spectrum_centre
        self._spectrum_sigma = spectrum_sigma
        self._min_sigma = min_sigma
        self._cell_relative_sigma = cell_relative_sigma
        self._use_B_covariance = use_B_covariance
        if cell_error not in ("independent", "correlated"):
            raise ValueError(f"Unknown cell error model {cell_error}")
        self._cell_error = cell_error
        # The spectrum (centre, sigma) used for each beam, for reporting
        self.spectra = []

    def _spectra_by_beam(self, reflections):
        """The (centre, sigma) of the spectrum of each experiment"""
        beams = []
        for expt in self._experiments:
            if not any(expt.beam is b for b in beams):
                beams.append(expt.beam)
        beam_index = [
            next(i for i, b in enumerate(beams) if expt.beam is b)
            for expt in self._experiments
        ]

        spectra = []
        expt_ids = reflections["id"].as_numpy_array()
        wavelength = reflections["wavelength"].as_numpy_array()
        for i_beam, beam in enumerate(beams):
            centre, sigma = _laue_spectrum_centre_and_sigma(beam)
            if self._spectrum_source == "reflections":
                expts = [i for i, b in enumerate(beam_index) if b == i_beam]
                sel = np.isin(expt_ids, expts) & (wavelength > 0)
                if sel.sum() > 1:
                    centre, sigma = _robust_centre_and_sigma(wavelength[sel])
            if self._spectrum_centre is not None:
                centre = self._spectrum_centre
            if self._spectrum_sigma is not None:
                sigma = self._spectrum_sigma
            spectra.append((centre, max(sigma, self._min_sigma)))
        self.spectra = spectra
        return [spectra[i] for i in beam_index]

    def calculate_weights(self, reflections):
        reflections = super().calculate_weights(reflections)
        wx, wy, _ = reflections["xyzobs.mm.weights"].parts()

        if self._model == "assignment":
            correlated = self._cell_error == "correlated"
            # For a correlated cell error, the cell error is not part of the
            # variance of each reflection: the observed wavelengths follow the
            # current cell instead
            variances = laue_assigned_wavelength_variances(
                self._experiments,
                reflections,
                cell_relative_sigma=0.0 if correlated else self._cell_relative_sigma,
                use_B_covariance=False if correlated else self._use_B_covariance,
            )
            floor = self._min_sigma**2
            variances.set_selected(variances < floor, floor)
            wz = 1.0 / variances
            if correlated:
                # The factor 2 sin(theta_obs) = |s1/|s1| - s0/|s0||, from the
                # observed scattering directions
                reflections["wavelength_obs_factor"] = _laue_two_sin_theta_obs(
                    self._experiments, reflections
                )
                reflections["wavelength_obs"] = laue_wavelength_obs_from_model(
                    self._experiments, reflections
                )
            else:
                reflections["wavelength_obs"] = reflections["wavelength"].deep_copy()
        else:
            expt_spectra = self._spectra_by_beam(reflections)
            centre = flex.double(len(reflections))
            wz = flex.double(len(reflections))
            ids = reflections["id"]
            for i_expt, (c, sigma) in enumerate(expt_spectra):
                sel = ids == i_expt
                centre.set_selected(sel, c)
                wz.set_selected(sel, 1.0 / sigma**2)
            reflections["wavelength_obs"] = centre

        reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, wz)
        return reflections
