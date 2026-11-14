"""Turning voxel masks into smooth, correctly placed surface meshes."""
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def brain_mask(images: dict) -> np.ndarray:
    """BraTS scans are skull-stripped, so any non-zero voxel is brain."""
    mask = np.zeros(next(iter(images.values())).shape, dtype=bool)
    for vol in images.values():
        mask |= vol > 0
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    return largest_component(mask)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def mask_to_mesh(
    mask: np.ndarray,
    affine: np.ndarray,
    smooth_sigma: float = 1.0,
    taubin_iters: int = 10,
    target_faces: int | None = 60_000,
    min_voxels: int = 20,
) -> trimesh.Trimesh | None:
    """Marching cubes on a lightly blurred mask, placed in scanner (mm) coordinates.

    Unlike the original notebook, no slices are dropped, so every mesh made from
    the same case shares one coordinate system and lines up exactly.
    """
    if mask.sum() < min_voxels:
        return None

    field = np.pad(mask.astype(np.float32), 2)
    if smooth_sigma > 0:
        blurred = ndimage.gaussian_filter(field, smooth_sigma)
        # a region made of scattered specks can blur away entirely (nothing left above 0.5);
        # then build the surface from the unblurred voxels instead of losing the region
        if blurred.max() > 0.5:
            field = blurred
    if field.max() <= 0.5:  # nothing to build a surface around
        return None
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5)
    verts -= 2  # undo padding

    # voxel index -> world millimetres
    verts = verts @ affine[:3, :3].T + affine[:3, 3]
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]

    mesh = trimesh.Trimesh(verts, faces, process=True)
    if taubin_iters:
        trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iters)
    if target_faces and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            pass  # fast_simplification not installed: keep full resolution
    mesh.fix_normals()
    return mesh
