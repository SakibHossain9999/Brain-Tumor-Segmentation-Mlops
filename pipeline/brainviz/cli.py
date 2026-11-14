"""Command line.

    brainviz msd BRATS_001 BRATS_002 --root data/msd/Task01_BrainTumour --out output/bundles
    brainviz brats path/to/case_folder --out output/bundles
"""
import argparse
from pathlib import Path

from .build import build_brats_case, build_msd_case


def main(argv=None):
    p = argparse.ArgumentParser(prog="brainviz", description="Brain tumour MRI + segmentation -> 3D viewer bundles")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("msd", help="MSD Task01 cases by id (ground-truth labels)")
    m.add_argument("cases", nargs="+", help="case ids, e.g. BRATS_001")
    m.add_argument("--root", type=Path, default=Path("data/msd/Task01_BrainTumour"))

    b = sub.add_parser("brats", help="BraTS case folders (one file per scan)")
    b.add_argument("cases", nargs="+", type=Path)

    for s in (m, b):
        s.add_argument("--out", type=Path, default=Path("output/bundles"), help="one subfolder per case")
        s.add_argument("--slice-modality", default="t2f", choices=["t1n", "t1c", "t2w", "t2f"])
        s.add_argument("--brain-faces", type=int, default=80_000)
        s.add_argument("--tumour-faces", type=int, default=30_000)

    args = p.parse_args(argv)
    opts = dict(slice_modality=args.slice_modality, brain_faces=args.brain_faces, tumour_faces=args.tumour_faces)
    for case in args.cases:
        if args.cmd == "msd":
            build_msd_case(case, args.root, args.out / case, **opts)
        else:
            build_brats_case(case, args.out / Path(case).name, **opts)


if __name__ == "__main__":
    main()
