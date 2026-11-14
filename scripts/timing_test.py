"""Timing test: how fast does each model train and predict on this GPU?

Uses random tensors shaped like real data, so it measures GPU speed only
(data loading is measured separately when we build the real training loop).
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `ml` can be imported
from ml.models import MODEL_NAMES, build_model  # noqa: E402


def sync():
    torch.cuda.synchronize()  # GPU work is asynchronous: wait for it before reading the clock


def time_model(name, args, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(name, use_checkpoint=args.swin_checkpoint).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = DiceCELoss(sigmoid=True)

    size = (args.crop,) * 3
    x = torch.randn(args.batch_size, 4, *size, device=device)
    y = (torch.rand(args.batch_size, 3, *size, device=device) > 0.9).float()

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):  # mixed precision
            out = model(x)
        loss = loss_fn(out.float(), y)
        loss.backward()
        opt.step()

    model.train()
    for _ in range(args.warmup):  # first steps are slow (memory setup, kernel tuning)
        train_step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    sync()
    s_per_step = (time.perf_counter() - t0) / args.steps
    train_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # validation-style prediction on one full-size scan (240 x 240 x 155)
    model.eval()
    vol = torch.randn(1, 4, 240, 240, 155, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sliding_window_inference(vol, size, 4, model, overlap=0.5)  # warm-up
        sync()
        t0 = time.perf_counter()
        sliding_window_inference(vol, size, 4, model, overlap=0.5)
        sync()
    s_per_case = time.perf_counter() - t0

    del model, opt, x, y, vol
    return {"params_m": params_m, "s_per_step": s_per_step, "train_mem_gb": train_mem_gb, "s_per_case": s_per_case}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--swin-checkpoint", action="store_true", help="gradient checkpointing for Swin UNETR (less memory, slower)")
    p.add_argument("--val-cases", type=int, default=72)
    p.add_argument("--val-every", type=int, default=1000)
    args = p.parse_args()

    assert torch.cuda.is_available(), "No GPU found"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | batch {args.batch_size} | crop {args.crop}^3\n")

    results = {}
    for name in args.models:
        print(f"Timing {name} ...", flush=True)
        try:
            results[name] = time_model(name, args, device)
        except torch.OutOfMemoryError:
            print(f"  {name}: OUT OF GPU MEMORY (try --swin-checkpoint or --batch-size 1)")
            torch.cuda.empty_cache()

    print(f"\n{'model':<12} {'params':>8} {'s/step':>7} {'train mem':>10} {'s/val case':>11}")
    for name, r in results.items():
        print(f"{name:<12} {r['params_m']:>7.1f}M {r['s_per_step']:>7.3f} {r['train_mem_gb']:>8.1f}GB {r['s_per_case']:>11.2f}")

    print("\nEstimated hours per training run (training + validation):")
    budgets = (20_000, 30_000, 40_000)
    print(f"{'model':<12} " + " ".join(f"{b // 1000:>5}k steps" for b in budgets))
    for name, r in results.items():
        hours = []
        for b in budgets:
            val_s = (b // args.val_every) * args.val_cases * r["s_per_case"]
            hours.append((b * r["s_per_step"] + val_s) / 3600)
        print(f"{name:<12} " + " ".join(f"{h:>9.1f} h " for h in hours))


if __name__ == "__main__":
    main()
