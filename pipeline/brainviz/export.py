"""Writing a case bundle the web viewer can load.

Bundle layout (one folder per case):
    brain.glb               the brain surface (mm, RAS), shared by every segmentation
    seg_<source>.glb        tumour zones from one source: "expert" (ground truth) or a model name;
                            one mesh per zone (edema, non_enhancing, enhancing)
    diff_<source>.glb       where a model disagrees with the expert, per region:
                            <REGION>_missed (expert only) and <REGION>_extra (model only)
    volume_<scan>.bin.gz    one file per MRI scan (t2f, t1n, t1c, t2w): uint8, cropped to the
                            brain's bounding box, x varies fastest, gzip-compressed (~1.5 MB each)
    meta.json               geometry, scan info, and per source: zone volumes, colours, scores
"""
import gzip
import json
from pathlib import Path

import numpy as np
import trimesh

from .labels import ZONES

BUNDLE_VERSION = 3


def normalise_to_uint8(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scan brightness -> 0..255 for display (robust to outliers: 0.5th..99.5th percentile)."""
    vals = vol[mask]
    lo, hi = (np.percentile(vals, [0.5, 99.5]) if vals.size else (0.0, 1.0))
    out = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    out[~mask] = 0
    return out.astype(np.uint8)


def write_glb(path, meshes: dict) -> list:
    """Named meshes -> one .glb (with normals: the viewer's shaders need them). Returns the names written."""
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        if mesh is not None:
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    if scene.geometry:
        scene.export(path, include_normals=True)
    return list(scene.geometry)


def write_volumes(out_dir: Path, scans: dict, offset, default_scan) -> dict:
    """`scans`: {"t2f": uint8 (X, Y, Z), ...}, all cropped to the same box starting at voxel `offset`."""
    files = {}
    for name, vol in scans.items():
        fname = f"volume_{name}.bin.gz"
        # three.js Data3DTexture expects x to vary fastest -> Fortran order
        with gzip.open(out_dir / fname, "wb", compresslevel=6) as f:
            f.write(np.ascontiguousarray(vol).ravel(order="F").tobytes())
        files[name] = fname
    return {
        "files": files,
        "shape": list(next(iter(scans.values())).shape),
        "offset": [int(o) for o in offset],  # voxel index of the crop's first corner
        "default": default_scan if default_scan in files else next(iter(files)),
        "dtype": "uint8",
        "encoding": "gzip",
    }


def zone_info(zone_voxels: dict, present: dict, voxel_ml: float) -> dict:
    return {
        key: {
            "name": info["name"],
            "color": info["color"],
            "volume_ml": round(zone_voxels.get(key, 0) * voxel_ml, 2),
            "present": present.get(key, False),
        }
        for key, info in ZONES.items()
    }


def clear_bundle(out_dir: Path):
    """Remove files of older bundle formats (and of sources no longer included)."""
    for pattern in ("scene.glb", "volume.bin", "seg_*.glb", "diff_*.glb", "volume_*.bin.gz"):
        for f in out_dir.glob(pattern):
            f.unlink()
