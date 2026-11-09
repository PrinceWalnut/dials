from __future__ import annotations

import logging
import warnings

import gemmi
import numpy as np
from scipy.spatial.transform import Rotation

import iotbx.phil
from dxtbx.model import Crystal
from libtbx import Auto
from scitbx import matrix

from dials.algorithms.indexing import DialsIndexError
from dials.array_family import flex

from .strategy import Strategy

logger = logging.getLogger(__name__)

pink_indexer_phil_str = """
pink_indexer
    .expert_level = 1
{
    max_refls = 50
        .type = int(value_min=10)
        .help = "Maximum number of reflections to consider indexing"
    wavelength = None
        .type = float(value_min=0.)
        .help = "The peak wavelength"
    percent_bandwidth = 1.
        .type = float(value_min=0.)
        .help = "The percent bandwidth used to calculate the wavelength range for indexing. The wavelength range is defined (wavelength - wavelength*percent_bandwidth/200, wavelength + wavelength*percent_bandwidth/200). This parameter also reflects the uncertainty of the supplied cell constants with larger values appropriate for less certain unit cells."
    rotogram_grid_points = 180
        .type = int(value_min=10, value_max=1000)
        .help = "Number of points at which to evaluate the angle search for each rlp-observation pair"
    voxel_grid_points=150
        .type = int(value_min=10, value_max=1000)
        .help = "Controls the number of voxels onto which the rotograms are discretized"
    target_lattices=1
        .type = int(value_min=1, value_max=100)
        .help = "The target number of candidate lattices to generate."
    min_lattices=1
        .type = int(value_min=1, value_max=100)
        .help = "(Deprecated) The minimum number of candidate lattices to generate."
    backend = *numpy torch
        .type = choice
        .help = "The array library used to evaluate and discretize the rotograms. "
                "torch requires PyTorch, and can run on a GPU (see device)."
    device = Auto
        .type = str
        .help = "The PyTorch device used with backend=torch, for instance cpu, cuda, "
                "cuda:1 or mps. Auto uses a CUDA or MPS GPU if one is available, and "
                "the CPU otherwise."
    max_rotations_per_batch = Auto
        .type = int(value_min=1)
        .help = "The maximum number of rotogram points evaluated at once, by "
                "default 4000000 (1000000 with backend=torch on the CPU). Memory "
                "use for the rotograms is roughly 100 bytes per point, on top of "
                "4*(voxel_grid_points+1)^3 bytes for the voxel grid. With "
                "backend=torch on a GPU, the batch is halved and retried if the GPU "
                "runs out of memory."
}
"""


def norm2(array, axis=-1, keepdims=False):
    """Faster version of np.linalg.norm for the L2 norm in lower dimensions."""
    a2 = np.square(array)
    out = 0.0
    for vec in np.split(a2, a2.shape[axis], axis=axis):
        out += vec
    out = np.sqrt(out)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def normalize(array, axis=-1):
    """Normalize a numpy array along a particular axis by dividing by its L2 norm"""
    out = array / norm2(array, axis=axis, keepdims=True)
    return out


def angle_between(vec1, vec2, deg=True):
    """
    This function computes the angle between vectors along the last dimension of the input arrays.
    This version is a numerically stable one based on arctan2 as described in this post:
     - https://scicomp.stackexchange.com/a/27769/39858

    Parameters
    ----------
    vec1 : array
        An arbitrarily batched arry of vectors
    vec2 : array
        An arbitrarily batched arry of vectors
    deg : bool (optional)
        Whether angles are returned in degrees or radians. The default is degrees (deg=True).

    Returns
    -------
    angles : array
        A vector of angles with the same leading dimensions of vec1 and vec2.
    """
    v1 = normalize(vec1, axis=-1)
    v2 = normalize(vec2, axis=-1)
    x1 = norm2(v1 - v2, axis=-1)
    x2 = norm2(v1 + v2, axis=-1)
    alpha = 2.0 * np.arctan2(x1, x2)
    if deg:
        return np.rad2deg(alpha)
    return alpha


def generate_reciprocal_cell(cell, dmin, dtype=np.int32):
    """
    Generate the miller indices of the full P1 reciprocal cell.

    Parameters
    ----------
    cell : tuple, list, np.ndarray of cell parameters, or gemmi.UnitCell
        Unit cell parameters
    dmin : float
        Maximum resolution of the data in Å
    dtype : np.dtype (optional)
        The data type of the returned array. The default is np.int32.


    Returns
    -------
    hkl : np.array(int32)
    """
    hmax, kmax, lmax = cell.get_hkl_limits(dmin)
    hkl = np.meshgrid(
        np.linspace(-hmax, hmax + 1, 2 * hmax + 2, dtype=dtype),
        np.linspace(-kmax, kmax + 1, 2 * kmax + 2, dtype=dtype),
        np.linspace(-lmax, lmax + 1, 2 * lmax + 2, dtype=dtype),
    )
    hkl = np.stack(hkl).reshape((3, -1)).T

    # Remove reflection 0,0,0
    hkl = hkl[np.any(hkl != 0, axis=1)]

    # Remove reflections outside of resolution range
    dHKL = cell.calculate_d_array(hkl).astype("float32")
    hkl = hkl[dHKL >= dmin]

    return hkl


def scaled_rotogram_rotvecs(hhat, qhat, cos_half_phi, sin_half_phi, xp=np):
    """
    Evaluate the rotograms for pairs of crystal-frame and lab-frame unit vectors.

    Each rotogram point is a rotation by pi about mhat, the bisector of hhat and
    qhat, which maps hhat onto qhat, followed by a rotation by phi about qhat. As
    quaternions, Rm = (mhat, 0) and Rq = (sin(phi/2) qhat, cos(phi/2)), so
        Rq * Rm = (cos(phi/2) mhat + sin(phi/2) qhat x mhat, -sin(phi/2) qhat.mhat)
    which needs only per-pair cross and dot products.

    Parameters
    ----------
    hhat : array
        An n x 3 array of unit vectors in the crystal frame
    qhat : array
        An n x 3 array of unit vectors in the lab frame
    cos_half_phi, sin_half_phi : array
        Cosines and sines of half the rotogram angles, of length m
    xp : module (optional)
        The array library of the inputs, numpy (the default) or torch.

    Returns
    -------
    scaled_rotvec : array
        An n x m x 3 array of rotation vectors, each of angle theta in [0, pi],
        rescaled to length arctan(theta / 4).
    """
    m = hhat + qhat
    mhat = m / xp.sqrt((m * m).sum(-1))[:, None]
    qx, qy, qz = qhat[:, 0], qhat[:, 1], qhat[:, 2]
    mx, my, mz = mhat[:, 0], mhat[:, 1], mhat[:, 2]
    qxm = xp.stack((qy * mz - qz * my, qz * mx - qx * mz, qx * my - qy * mx), -1)
    xyz = (
        cos_half_phi[None, :, None] * mhat[:, None, :]
        + sin_half_phi[None, :, None] * qxm[:, None, :]
    )
    w = -sin_half_phi[None, :] * (qhat * mhat).sum(-1)[:, None]

    # Taking |w| picks the quaternion sign giving theta in [0, pi]
    sin_half_theta = xp.sqrt((xyz * xyz).sum(-1))
    theta = 2.0 * xp.arctan2(sin_half_theta, xp.abs(w))

    # Scaling the general rotvec norm by this factor makes the discretization more uniform
    # without this, the rotations would be biased toward higher angles
    scale = xp.arctan(theta / 4.0) / sin_half_theta
    scale = xp.where(w < 0, -scale, scale)
    return xyz * scale[..., None]


def rotogram_voxels_numpy(hhat, qhat, phi, bins, max_rotations_per_batch=None):
    """
    Count the rotogram points of each pair of hhat and qhat falling in each voxel
    of the grid with the given (uniformly spaced) bin edges along each axis,
    using numpy.
    """
    n = len(bins) + 1
    bin_min = float(bins[0])
    inv_bin_width = (len(bins) - 1) / float(bins[-1] - bins[0])
    voxels = np.zeros(n**3, dtype=np.int32)
    cos_half_phi, sin_half_phi = np.cos(phi / 2), np.sin(phi / 2)
    # Larger batches amortize the cost of np.bincount over the whole voxel grid
    max_rotations_per_batch = max_rotations_per_batch or 4000000
    batch = max(1, max_rotations_per_batch // len(phi))
    for start in range(0, len(hhat), batch):
        rotvecs = scaled_rotogram_rotvecs(
            hhat[start : start + batch],
            qhat[start : start + batch],
            cos_half_phi,
            sin_half_phi,
        )
        # Equivalent to np.digitize(rotvecs, bins), without a binary search.
        # fmin and fmax put NaNs in the last bin, as np.digitize does.
        idx = np.floor((rotvecs - bin_min) * inv_bin_width)
        idx = np.fmax(np.fmin(idx, n - 2), -1).astype(np.int64) + 1
        flat_idx = (idx[..., 0] * n + idx[..., 1]) * n + idx[..., 2]
        voxels += np.bincount(flat_idx.ravel(), minlength=n**3).astype(np.int32)
    return voxels.reshape((n, n, n))


def torch_device(device="auto"):
    """
    The torch.device for a device name, where auto chooses a CUDA or MPS GPU if
    one is available, and the CPU otherwise.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return device


def rotogram_voxels_torch(
    hhat, qhat, phi, bins, max_rotations_per_batch=None, device="auto"
):
    """
    Count the rotogram points of each pair of hhat and qhat falling in each voxel
    of the grid with the given (uniformly spaced) bin edges along each axis,
    using torch on the given device. If the device runs out of memory, the
    batch is halved and retried.
    """
    import torch

    device = torch_device(device)
    hhat, qhat, phi = (
        torch.as_tensor(np.ascontiguousarray(a), device=device)
        for a in (hhat, qhat, phi)
    )
    n = len(bins) + 1
    bin_min = float(bins[0])
    inv_bin_width = (len(bins) - 1) / float(bins[-1] - bins[0])
    voxels = torch.zeros(n**3, dtype=torch.int32, device=device)
    one = torch.ones(1, dtype=torch.int32, device=device)
    cos_half_phi, sin_half_phi = torch.cos(phi / 2), torch.sin(phi / 2)
    # Smaller batches are faster on the CPU, as they stay in cache, and larger
    # batches are faster on a GPU, needing fewer kernel launches
    if not max_rotations_per_batch:
        max_rotations_per_batch = 1000000 if device.type == "cpu" else 4000000
    batch = max(1, max_rotations_per_batch // len(phi))
    start = 0
    while start < len(hhat):
        try:
            rotvecs = scaled_rotogram_rotvecs(
                hhat[start : start + batch],
                qhat[start : start + batch],
                cos_half_phi,
                sin_half_phi,
                xp=torch,
            )
            # Equivalent to torch.bucketize(rotvecs, bins, right=True), without
            # a binary search, with NaNs in the last bin
            idx = torch.floor((rotvecs - bin_min) * inv_bin_width)
            idx = idx.nan_to_num_(nan=n - 2).clamp_(-1, n - 2).to(torch.int64).add_(1)
            del rotvecs
            flat_idx = ((idx[..., 0] * n + idx[..., 1]) * n + idx[..., 2]).ravel()
            del idx
            voxels.scatter_add_(0, flat_idx, one.expand(len(flat_idx)))
            del flat_idx
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            logger.debug(
                "Out of memory on %s, retrying with %d rotograms per batch",
                device,
                batch,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        start += batch
    return voxels.reshape((n, n, n)).cpu().numpy()


def top_voxels_cutoff(voxels, target):
    """
    The voxel count above which lie the target number of highest-count voxels,
    or fewer if there is a tie at the cutoff. Equivalent to
    np.quantile(voxels, (voxels.size - target) / voxels.size) for an integer
    voxel grid, but computed from a histogram of the counts.
    """
    if not np.issubdtype(voxels.dtype, np.integer):
        return np.quantile(voxels, (voxels.size - target) / voxels.size)
    # Number of voxels with a count >= each count value
    at_least = np.cumsum(np.bincount(voxels.ravel())[::-1])[::-1]
    # The target'th and (target + 1)'th highest counts
    kth = np.flatnonzero(at_least >= target)[-1]
    next_kth = np.flatnonzero(at_least >= target + 1)[-1]
    if kth == next_kth:
        return kth
    return 0.5 * (kth + next_kth)


class Indexer:
    """
    A class which implements the PinkIndexer algorithm.
    [Gevorkov Y, et al. pinkIndexer – a universal indexer for pink-beam X-ray and electron diffraction snapshots. Acta Cryst A. 2020 Mar 1;76(2):121–31.](https://doi.org/10.1107/S2053273319015559)
    """

    def __init__(self, cell, wavelength, bandwidth, float_dtype="float32"):
        """
        Args:
            cell (gemmi.UnitCell): The target cell.
            wavelength (float): Peak wavelength in Å.
            bandwidth (float): The percentage bandwidth used to calculate a wavelength range
        """
        self.float_dtype = float_dtype
        self.cell = cell
        self.wav_peak = wavelength
        self.wav_min = wavelength - bandwidth * wavelength / 200.0
        self.wav_max = wavelength + bandwidth * wavelength / 200.0
        self.B = np.asarray(cell.frac.mat, dtype=self.float_dtype).T

    def index_pink(
        self,
        rlps,
        max_refls=50,
        rotogram_grid_points=200,
        voxel_grid_points=200,
        float_dtype="float32",
        target_lattices=1,
        dilate_r=None,
        backend="numpy",
        device="auto",
        max_rotations_per_batch=None,
    ):
        """
        Args:
            rlps (array): An n x 3 array floating point numbers corresponding to the rlp vectors. This will be normalized.
            max_refls (int, optional): The maximum number of refls to consider.
            rotogram_grid_points (int, optional): The number of points at which to evaluate the rotograms.
            voxel_grid_points (int, optional): The fineness of the discretization used to quantitate rotograms.
            float_dtype (str or dtype, optional): the dtype to use for floating point values
            target_lattices (int, optional): the target number of candidate lattices returned by this function
            dilate_r (float, optional): optionally dilate the voxel grid by a kernel with this radius in pixels.
            backend (str, optional): evaluate the rotograms with "numpy" or "torch".
            device (str, optional): the torch device for backend="torch", or "auto".
            max_rotations_per_batch (int, optional): the maximum number of rotogram points evaluated at once, or None for a default for the backend.

        Returns:
            iter_UB: an interable of crystal bases as numpy arrays
        """
        q = rlps
        q_len = norm2(q, keepdims=True)  # length of q if lambda is 1A

        # Possible resolution range for each observation considering wavelength range
        res_min = self.wav_min / q_len.squeeze(-1)
        res_max = self.wav_max / q_len.squeeze(-1)

        in_range = None
        # Truncate the resolution range if there are too many reflections
        if len(res_min) > max_refls:
            dmin = np.sort(res_min)[-max_refls]
            in_range = res_min >= dmin
            q = q[in_range]
            q_len = q_len[in_range]
            res_min = res_min[in_range]
            res_max = res_max[in_range]
        else:
            dmin = res_min.min()

        # Normalized q vector
        qhat = normalize(q)

        # Generate the feasible set of reflections from the current geometry
        # These are in cartesian reciprocal space coordinates in the
        # crystal-fixed system
        Hall = generate_reciprocal_cell(self.cell, dmin, dtype=self.float_dtype)
        h = (self.B @ Hall.T).T
        hhat = normalize(h)

        # Remove candidate rlps if incompatible with resolution range of observation
        dall = self.cell.calculate_d_array(Hall)
        mask = (dall <= res_max[:, None]) & (dall >= res_min[:, None])
        i, j = np.where(mask)

        hhat = hhat[j]
        qhat = qhat[i]

        # Rotogram angles about qhat
        phimin = -np.pi
        phimax = np.pi
        phi = np.linspace(
            phimin, phimax, rotogram_grid_points + 1, dtype=self.float_dtype
        )[:-1]

        # Bin edges for discretizing the scaled rotvecs
        scale_max = np.arctan(np.pi / 4.0)
        bins = np.linspace(-scale_max, scale_max, voxel_grid_points)

        # This is how to calculate the bin centers to extract the rotation matrix from the voxel grid
        bin_centers = np.concatenate(
            (bins[[0]], 0.5 * (bins[1:] + bins[:-1]), bins[[-1]])
        )

        # Map discretized rotvecs for all rotograms into voxel grid
        bins = bins.astype(self.float_dtype)
        if backend == "torch":
            voxels = rotogram_voxels_torch(
                hhat, qhat, phi, bins, max_rotations_per_batch, device=device
            )
        else:
            voxels = rotogram_voxels_numpy(
                hhat, qhat, phi, bins, max_rotations_per_batch
            )

        # Optionally dilate the voxel grid
        if dilate_r is not None:
            # PinkIndexer does a little dilation to help avoid overfitting
            # I'm implementing this using a radially symmetric convolution
            # In the original paper, they use a cubic kernel ones((3, 3, 3))
            voxels = voxels.astype(float_dtype)
            x = np.arange(voxels.shape[0], dtype=float_dtype)
            x = np.square(x - x.mean())
            kernel = np.exp(
                -(x[:, None, None] + x[None, :, None] + x[None, None, :])
                / np.square(dilate_r)
            )
            from scipy.signal import fftconvolve

            voxels = fftconvolve(voxels, kernel, mode="same")

        # Possible solutions are voxels with the highest density
        cutoff = top_voxels_cutoff(voxels, target_lattices)
        idx = np.where(voxels > cutoff)
        asort = np.argsort(voxels[idx])
        sortidx = tuple(i[asort[::-1]] for i in idx)  # Descending order
        peaks = np.column_stack(sortidx)

        # Get UB matrices for best peak voxels
        for peak in peaks:
            v = bin_centers[peak]
            l = norm2(v)
            theta = np.tan(l) * 4.0
            rotvec = theta * v / l
            U = Rotation.from_rotvec(rotvec).as_matrix()
            UB = U @ self.B
            yield UB


class PinkIndexer(Strategy):
    """
    A lattice search strategy using the pinkIndexer algorithm.

    For more info, see:
    [Gevorkov Y, et al. pinkIndexer – a universal indexer for pink-beam X-ray and electron diffraction snapshots. Acta Cryst A. 2020 Mar 1;76(2):121–31.](https://doi.org/10.1107/S2053273319015559)
    """

    phil_help = (
        "A lattice search strategy that matches low resolution spots to candidate "
        "indices based on a known unit cell and space group. It supports mono and "
        "polychromatic beams. "
    )

    phil_scope = iotbx.phil.parse(pink_indexer_phil_str)

    def __init__(
        self, target_symmetry_primitive, max_lattices, params=None, *args, **kwargs
    ):
        """Construct PinkIndexer object.

        Args:
            target_symmetry_primitive (cctbx.crystal.symmetry): The target
                crystal symmetry and unit cell

            max_lattices (int): The maximum number of lattice models to find

            params (phil,optional): Phil params
        """
        super().__init__(params=None, *args, **kwargs)
        self._target_symmetry_primitive = target_symmetry_primitive
        self._max_lattices = max_lattices

        if target_symmetry_primitive is None:
            raise DialsIndexError(
                "Target unit cell and space group must be provided for pink_indexer"
            )

        target_cell = target_symmetry_primitive.unit_cell()
        if target_cell is None:
            raise ValueError("Please specify known_symmetry.unit_cell")

        self.cell = target_cell
        self.tarsym = target_symmetry_primitive
        self.spacegroup = target_symmetry_primitive.space_group()
        self.wavelength = params.wavelength
        self.percent_bandwidth = params.percent_bandwidth
        self.max_refls = params.max_refls
        self.rotogram_grid_points = params.rotogram_grid_points
        self.voxel_grid_points = params.voxel_grid_points
        if params.min_lattices != 1:
            warnings.warn(
                "The parameter min_lattices is deprecated. Please use target_lattices instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            params.target_lattices = params.min_lattices
        self.target_lattices = params.target_lattices
        self.backend = params.backend
        self.device = "auto" if params.device in (None, Auto) else params.device
        self.max_rotations_per_batch = params.max_rotations_per_batch
        if self.max_rotations_per_batch is Auto:
            self.max_rotations_per_batch = None
        if self.backend == "torch":
            try:
                torch_device(self.device)
            except ImportError:
                raise DialsIndexError(
                    "pink_indexer.backend=torch requires PyTorch, which is not installed"
                )
            except RuntimeError as e:
                raise DialsIndexError(f"Invalid pink_indexer.device={self.device}: {e}")

    def find_crystal_models(self, reflections, experiments):
        """Find a list of candidate crystal models.
        Args:
            reflections (dials.array_family.flex.reflection_table):
                The found spots centroids and associated data
            experiments (dxtbx.model.experiment_list.ExperimentList):
                The experimental geometry models
        """
        # This is a workaround for https://github.com/dials/dials/issues/2485
        reflections["id"] *= 0
        reflections["id"] -= 1

        # Get data from still
        expt = experiments[0]
        refls = reflections

        # Build unit cell
        cell = gemmi.UnitCell(*self.cell.parameters())

        # Get wavelength and bandwidth
        wav, bw = self.wavelength, self.percent_bandwidth
        beam = expt.beam
        if wav is None:
            if experiments.all_laue() or experiments.all_tof():
                assert "wavelength" in reflections
                wav = reflections["wavelength"]
            else:
                wav = beam.get_wavelength()

        if bw is None:
            bw = 1.0
        pidxr = Indexer(cell, wav, bw)

        # Get reciprocal lattice points and beam info
        rlps = flex.vec3_double(len(refls))
        s1 = refls["s1"]
        s1_hat = s1 / s1.norms()
        s0_hat = beam.get_unit_s0()

        # Rotate reflections to common coordinate frame
        for i, expt in enumerate(experiments):
            sel_expt = refls["imageset_id"] == i
            s1_hat_expt = s1_hat.select(sel_expt)
            q = s1_hat_expt - s0_hat

            if expt.goniometer is not None:
                setting_rotation = matrix.sqr(expt.goniometer.get_setting_rotation())
                rotation_axis = expt.goniometer.get_rotation_axis_datum()
                sample_rotation = matrix.sqr(expt.goniometer.get_fixed_rotation())
                q = tuple(setting_rotation.inverse()) * q

                if expt.scan is not None and expt.scan.has_property("oscillation"):
                    _, _, z = refls["xyzobs.mm.value"].select(sel_expt).parts()
                    q.rotate_around_origin(rotation_axis, -z)

                q = tuple(sample_rotation.inverse()) * q

            rlps.set_selected(sel_expt, q)

        rlps = np.array(rlps, dtype="float32")
        self.candidate_crystal_models = []

        # Get candidate crystal models from lattice search
        for UB in pidxr.index_pink(
            rlps,
            self.max_refls,
            rotogram_grid_points=self.rotogram_grid_points,
            voxel_grid_points=self.voxel_grid_points,
            target_lattices=self.target_lattices,
            backend=self.backend,
            device=self.device,
            max_rotations_per_batch=self.max_rotations_per_batch,
        ):
            real_a, real_b, real_c = np.linalg.inv(UB.astype("double"))
            crystal = Crystal(real_a, real_b, real_c, self.spacegroup)
            self.candidate_crystal_models.append(crystal)
        return self.candidate_crystal_models
