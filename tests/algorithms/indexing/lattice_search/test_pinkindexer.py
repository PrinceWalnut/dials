from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dials.algorithms.indexing.lattice_search import pinkindexer


def random_unit_vectors(n, rng):
    v = rng.normal(size=(n, 3)).astype("float32")
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def rotogram_inputs():
    rng = np.random.default_rng(42)
    hhat = random_unit_vectors(500, rng)
    qhat = random_unit_vectors(500, rng)
    phi = np.linspace(-np.pi, np.pi, 37, dtype="float32")[:-1]
    scale_max = np.arctan(np.pi / 4.0)
    bins = np.linspace(-scale_max, scale_max, 30).astype("float32")
    return hhat, qhat, phi, bins


def reference_scaled_rotvecs(hhat, qhat, phi):
    """A rotation by pi about the bisector of hhat and qhat, then by phi about qhat"""
    mhat = hhat + qhat
    mhat /= np.linalg.norm(mhat, axis=-1, keepdims=True)
    Rm = Rotation.from_rotvec(np.pi * mhat.astype("float64"))
    out = np.empty((len(hhat), len(phi), 3))
    for i in range(len(hhat)):
        Rq = Rotation.from_rotvec(qhat[i].astype("float64")[None, :] * phi[:, None])
        rotvec = (Rq * Rm[i]).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        out[i] = rotvec / theta * np.arctan(theta / 4.0)
    return out


def test_scaled_rotogram_rotvecs(rotogram_inputs):
    hhat, qhat, phi, _ = rotogram_inputs
    rotvecs = pinkindexer.scaled_rotogram_rotvecs(
        hhat, qhat, np.cos(phi / 2), np.sin(phi / 2)
    )
    assert rotvecs.shape == (len(hhat), len(phi), 3)
    assert rotvecs.dtype == np.float32
    expected = reference_scaled_rotvecs(hhat, qhat, phi)
    # At phi = 0, the rotation is by pi about mhat, which is the same rotation as
    # by pi about -mhat, so the sign of the rotvec is arbitrary
    at_pi = np.isclose(phi, 0, atol=1e-6)
    assert at_pi.sum() == 1
    np.testing.assert_allclose(rotvecs[:, ~at_pi], expected[:, ~at_pi], atol=1e-5)
    np.testing.assert_allclose(
        np.abs(rotvecs[:, at_pi]), np.abs(expected[:, at_pi]), atol=1e-5
    )

    # Every rotation maps hhat onto qhat
    theta = 4.0 * np.tan(np.linalg.norm(rotvecs, axis=-1))
    axes = rotvecs / np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    R = Rotation.from_rotvec((axes * theta[..., None]).reshape(-1, 3))
    np.testing.assert_allclose(
        R.apply(np.repeat(hhat, len(phi), axis=0)),
        np.repeat(qhat, len(phi), axis=0),
        atol=1e-4,
    )


@pytest.mark.parametrize("max_rotations_per_batch", [None, 1, 1000])
def test_rotogram_voxels_numpy(rotogram_inputs, max_rotations_per_batch):
    hhat, qhat, phi, bins = rotogram_inputs
    voxels = pinkindexer.rotogram_voxels_numpy(
        hhat, qhat, phi, bins, max_rotations_per_batch
    )
    n = len(bins) + 1
    assert voxels.shape == (n, n, n)
    assert voxels.sum() == len(hhat) * len(phi)

    # Compare with np.digitize of the reference rotvecs, away from phi = 0 where
    # the rotvec sign is arbitrary. Points within rounding error of a bin edge
    # may be binned on either side of it.
    phi = phi + 0.01
    voxels = pinkindexer.rotogram_voxels_numpy(
        hhat, qhat, phi, bins, max_rotations_per_batch
    )
    idx = np.digitize(reference_scaled_rotvecs(hhat, qhat, phi), bins)
    expected = np.zeros((n, n, n), dtype=int)
    np.add.at(expected, tuple(idx.transpose((2, 0, 1))), 1)
    assert np.abs(voxels - expected).sum() <= 0.001 * expected.sum()


def test_rotogram_voxels_torch(rotogram_inputs):
    pytest.importorskip("torch")
    hhat, qhat, phi, bins = rotogram_inputs
    expected = pinkindexer.rotogram_voxels_numpy(hhat, qhat, phi, bins)
    for max_rotations_per_batch in (None, 1000):
        voxels = pinkindexer.rotogram_voxels_torch(
            hhat, qhat, phi, bins, max_rotations_per_batch, device="cpu"
        )
        assert voxels.dtype == np.int32
        assert voxels.sum() == expected.sum()
        # Allowing for last-bit differences in transcendental functions
        assert np.abs(voxels - expected).sum() <= 0.001 * expected.sum()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rotogram_voxels_antiparallel(rotogram_inputs, backend):
    # The bisector of antiparallel hhat and qhat is undefined, giving NaN
    # rotvecs, which are counted in the last voxel as np.digitize would
    if backend == "torch":
        pytest.importorskip("torch")
    hhat, qhat, phi, bins = rotogram_inputs
    hhat = np.concatenate((hhat[:3], -qhat[3:4]))
    qhat = qhat[:4]
    if backend == "torch":
        voxels = pinkindexer.rotogram_voxels_torch(hhat, qhat, phi, bins, device="cpu")
    else:
        voxels = pinkindexer.rotogram_voxels_numpy(hhat, qhat, phi, bins)
    assert voxels.sum() == 4 * len(phi)
    assert voxels[-1, -1, -1] == len(phi)


def test_rotogram_voxels_torch_out_of_memory(rotogram_inputs, monkeypatch):
    torch = pytest.importorskip("torch")
    hhat, qhat, phi, bins = rotogram_inputs
    expected = pinkindexer.rotogram_voxels_torch(hhat, qhat, phi, bins, device="cpu")

    # Run out of memory for batches of more than 100 rotogram pairs
    scaled_rotogram_rotvecs = pinkindexer.scaled_rotogram_rotvecs
    batch_sizes = []

    def limited_scaled_rotogram_rotvecs(hhat, *args, **kwargs):
        batch_sizes.append(len(hhat))
        if len(hhat) > 100:
            raise torch.OutOfMemoryError
        return scaled_rotogram_rotvecs(hhat, *args, **kwargs)

    monkeypatch.setattr(
        pinkindexer, "scaled_rotogram_rotvecs", limited_scaled_rotogram_rotvecs
    )
    voxels = pinkindexer.rotogram_voxels_torch(
        hhat, qhat, phi, bins, 1000 * len(phi), device="cpu"
    )
    # Halving a batch of 1000 pairs first retries all 500 pairs
    assert batch_sizes[:5] == [500, 500, 250, 125, 62]
    assert max(batch_sizes[5:]) <= 62
    np.testing.assert_array_equal(voxels, expected)


def test_torch_device():
    torch = pytest.importorskip("torch")
    assert pinkindexer.torch_device("cpu") == torch.device("cpu")
    assert pinkindexer.torch_device("auto").type in ("cpu", "cuda", "mps")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            pinkindexer.torch_device("cuda")


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float64"])
def test_top_voxels_cutoff(dtype):
    rng = np.random.default_rng(0)
    voxels = rng.poisson(2, size=(20, 20, 20)).astype(dtype)
    voxels.flat[[10, 20, 30, 40, 50]] = [30, 25, 25, 20, 19]
    for target, n_expected in [(1, 1), (2, 1), (3, 3), (4, 4), (5, 5)]:
        cutoff = pinkindexer.top_voxels_cutoff(voxels, target)
        expected = np.quantile(voxels, (voxels.size - target) / voxels.size)
        assert (voxels > cutoff).sum() == (voxels > expected).sum() == n_expected


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_index_pink_backends(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    gemmi = pytest.importorskip("gemmi")
    cell = gemmi.UnitCell(34.3, 45.6, 99.0, 90, 90, 90)
    indexer = pinkindexer.Indexer(cell, wavelength=1.1, bandwidth=15.0)

    # Simulate the low resolution reflections in the wavelength range for a
    # random crystal orientation, as rlps scaled to a wavelength of 1 A
    U = Rotation.from_rotvec([0.3, -0.5, 1.1]).as_matrix()
    hkl = pinkindexer.generate_reciprocal_cell(cell, 5.0)
    s = (U @ indexer.B @ hkl.T).T
    s0 = np.array([0.0, 0.0, -1.0])
    wavelength = -2 * (s @ s0) / (s * s).sum(-1)
    s = s[(wavelength >= indexer.wav_min) & (wavelength <= indexer.wav_max)]
    wavelength = -2 * (s @ s0) / (s * s).sum(-1)
    rlps = (s * wavelength[:, None]).astype("float32")

    UBs = list(
        indexer.index_pink(
            rlps,
            max_refls=50,
            rotogram_grid_points=180,
            voxel_grid_points=150,
            target_lattices=3,
            backend=backend,
            device="cpu",
        )
    )
    assert 1 <= len(UBs) <= 3
    # The best candidate is within a voxel or two of the true orientation, up
    # to the 222 symmetry of the lattice
    U_found = UBs[0] @ np.linalg.inv(indexer.B)
    twofolds = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    misorientation = min(
        Rotation.from_matrix(U_found @ R @ U.T).magnitude() for R in twofolds
    )
    assert np.degrees(misorientation) < 2.0
