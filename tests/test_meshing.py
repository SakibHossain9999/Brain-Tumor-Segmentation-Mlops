"""Voxel masks -> 3D surfaces."""
import numpy as np

from brainviz.meshing import mask_to_mesh


def ball(shape=(48, 48, 48), centre=(24, 24, 24), radius=8):
    idx = np.indices(shape)
    return sum((i - c) ** 2 for i, c in zip(idx, centre)) <= radius**2


def test_a_ball_becomes_a_closed_surface_of_the_right_size():
    mesh = mask_to_mesh(ball(), np.eye(4), target_faces=None)
    assert mesh is not None and mesh.is_watertight
    size = mesh.bounds[1] - mesh.bounds[0]
    assert np.allclose(size, 16, atol=2)  # diameter 16 voxels = 16 mm at 1 mm spacing


def test_the_affine_places_the_surface_in_millimetres():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [100, 0, -50]
    mesh = mask_to_mesh(ball(), affine, target_faces=None)
    centre = mesh.bounds.mean(axis=0)
    assert np.allclose(centre, [100 + 48, 48, -50 + 48], atol=2)  # voxel 24 * 2 mm + offset


def test_an_empty_or_tiny_mask_gives_no_surface():
    assert mask_to_mesh(np.zeros((32, 32, 32), bool), np.eye(4)) is None
    tiny = np.zeros((32, 32, 32), bool)
    tiny[5, 5, 5] = True
    assert mask_to_mesh(tiny, np.eye(4)) is None  # below the minimum size


def test_scattered_specks_do_not_crash():
    # regression: specks spread far apart blur away completely (nothing above 0.5), which used to make
    # the surface builder raise "Surface level must be within volume data range"
    mask = np.zeros((64, 64, 64), bool)
    mask[::8, ::8, ::8] = True  # 512 isolated voxels
    mesh = mask_to_mesh(mask, np.eye(4), smooth_sigma=0.8, target_faces=None)
    assert mesh is not None  # built from the unblurred voxels instead
