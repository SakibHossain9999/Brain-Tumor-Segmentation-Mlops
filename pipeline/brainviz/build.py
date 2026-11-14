"""End-to-end: a loaded case + one or more segmentations -> viewer bundle.

A "segmentation source" is where a tumour outline comes from: the expert (ground truth) or a model.
    build_bundle(case, out)                                  # expert outline only
    build_bundle(case, out, {"expert": {...}, "segresnet": {"regions": ..., "label": "SegResNet"}})
When the expert outline is included, every model also gets a disagreement file (missed / extra tumour).
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .export import BUNDLE_VERSION, clear_bundle, normalise_to_uint8, write_glb, write_volumes, zone_info
from .io import Case, load_brats_case, load_msd_case
from .labels import REGION_NAMES, zones_from_regions
from .meshing import brain_mask, mask_to_mesh

DIFF_MIN_VOXELS = 10  # disagreement blobs smaller than this (thin slivers along edges) are not drawn


def remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop connected pieces smaller than `min_voxels`."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_voxels
    return keep[labels]


def write_segmentation(case: Case, out_dir: Path, source: str, seg: dict, expert, tumour_faces: int = 30_000) -> dict:
    """One segmentation's files: its tumour zones (seg_<source>.glb) and, if an expert outline is given and
    this is a model, where it disagrees with the expert (diff_<source>.glb). Returns its meta.json entry."""
    voxel_ml = float(np.prod(case.spacing)) / 1000.0  # 1 mm^3 voxels -> 0.001 mL
    regions = np.asarray(seg["regions"], bool)
    shape = next(iter(case.images.values())).shape
    if regions.shape != (3, *shape):
        raise ValueError(f"{source}: regions shape {regions.shape} does not match scan shape (3, {shape})")

    # the tumour zones the viewer draws
    zones = zones_from_regions(regions)
    meshes = {k: mask_to_mesh(m, case.affine, smooth_sigma=0.8, taubin_iters=10, target_faces=tumour_faces)
              for k, m in zones.items()}
    write_glb(out_dir / f"seg_{source}.glb", meshes)
    info = {
        "label": seg.get("label", source),
        "file": f"seg_{source}.glb",
        "zones": zone_info({k: int(m.sum()) for k, m in zones.items()},
                           {k: m is not None for k, m in meshes.items()}, voxel_ml),
        "regions_ml": {name: round(int(r.sum()) * voxel_ml, 2) for name, r in zip(REGION_NAMES, regions)},
    }
    info.update({k: v for k, v in seg.items() if k not in ("regions", "label")})

    # where a model disagrees with the expert, per region
    if expert is not None and source != "expert":
        expert = np.asarray(expert, bool)
        diff_meshes, diff_ml = {}, {}
        for i, name in enumerate(REGION_NAMES):
            for kind, mask in (("missed", expert[i] & ~regions[i]), ("extra", regions[i] & ~expert[i])):
                mask = remove_small(mask, DIFF_MIN_VOXELS)
                diff_ml[f"{name}_{kind}"] = round(int(mask.sum()) * voxel_ml, 2)
                diff_meshes[f"{name}_{kind}"] = mask_to_mesh(mask, case.affine, smooth_sigma=0.6,
                                                             taubin_iters=5, target_faces=tumour_faces)
        written = write_glb(out_dir / f"diff_{source}.glb", diff_meshes)
        info["diff"] = {"file": f"diff_{source}.glb" if written else None, "volume_ml": diff_ml,
                        "min_voxels": DIFF_MIN_VOXELS}
    return info


def add_segmentation(case: Case, out_dir: str | Path, source: str, seg: dict, tumour_faces: int = 30_000) -> dict:
    """Add one more segmentation (e.g. another model) to an existing bundle, without rebuilding the brain
    or the scans. The case's expert outline (case.regions), if any, is used for the comparison files."""
    out_dir = Path(out_dir)
    meta_file = out_dir / "meta.json"
    meta = json.loads(meta_file.read_text())
    info = write_segmentation(case, out_dir, source, seg, case.regions, tumour_faces)
    meta["segmentations"][source] = info
    meta_file.write_text(json.dumps(meta, indent=2))
    return info


def build_bundle(
    case: Case,
    out_dir: str | Path,
    segmentations: dict | None = None,
    default: str | None = None,
    slice_modality: str = "t2f",
    brain_faces: int = 80_000,
    tumour_faces: int = 30_000,
    extra_meta: dict | None = None,
    log=print,
) -> Path:
    """`segmentations`: {source: {"regions": (3, H, W, D) WT/TC/ET masks, "label": display name, ...extra}}.
    Extra keys (e.g. scores, model version) are copied into meta.json. Default: the case's expert outline.
    `slice_modality`: which scan the viewer shows first (all four are exported)."""
    if segmentations is None:
        if case.regions is None:
            raise ValueError(f"{case.case_id}: no expert outline and no segmentation given")
        segmentations = {"expert": {"regions": case.regions, "label": "Expert outline"}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_bundle(out_dir)

    brain = brain_mask(case.images)
    brain_mesh = mask_to_mesh(brain, case.affine, smooth_sigma=1.2, taubin_iters=15, target_faces=brain_faces)
    write_glb(out_dir / "brain.glb", {"brain": brain_mesh})

    expert = segmentations.get("expert", {}).get("regions")
    expert = None if expert is None else np.asarray(expert, bool)
    sources = {}
    for source, seg in segmentations.items():
        info = write_segmentation(case, out_dir, source, seg, expert, tumour_faces)
        sources[source] = info
        log(f"[{case.case_id}] {source}: " + ", ".join(f"{k} {v['volume_ml']} mL" for k, v in info["zones"].items()))

    # all four scans, cropped to the brain's bounding box (the black border carries no information)
    idx = np.nonzero(brain)
    lo = [int(i.min()) for i in idx]
    hi = [int(i.max()) + 1 for i in idx]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    scans = {name: normalise_to_uint8(vol, brain)[box] for name, vol in case.images.items()}

    meta = {
        "bundle_version": BUNDLE_VERSION,
        "case_id": case.case_id,
        "units": "mm",
        "spacing": list(case.spacing),
        "bounds": brain_mesh.bounds.tolist(),
        "center": brain_mesh.bounds.mean(axis=0).tolist(),
        "affine": case.affine.tolist(),
        "brain": "brain.glb",
        "volumes": write_volumes(out_dir, scans, lo, slice_modality),
        "segmentations": sources,
        "default_segmentation": default if default in sources else next(iter(sources)),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[{case.case_id}] wrote {out_dir} ({len(sources)} segmentations, {len(scans)} scans)")
    return out_dir


def build_msd_case(case_id, root, out_dir, **kwargs) -> Path:
    return build_bundle(load_msd_case(case_id, root), out_dir, **kwargs)


def build_brats_case(case_dir, out_dir, **kwargs) -> Path:
    return build_bundle(load_brats_case(case_dir), out_dir, **kwargs)
