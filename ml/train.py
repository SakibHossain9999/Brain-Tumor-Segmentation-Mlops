"""Train one model:  python ml/train.py --config ml/configs/segresnet.yaml

Everything except the model comes from ml/configs/base.yaml, so all models are trained identically.
Logs to MLflow (mlflow.db); checkpoints go to output/runs/<run name>/.
"""
import argparse
import csv
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data import REGIONS, get_loaders  # noqa: E402
from ml.models import build_model  # noqa: E402


# ---------------------------------------------------------------- config
def load_config(path):
    """Load a model config on top of base.yaml."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = yaml.safe_load((path.parent / cfg.pop("base")).read_text())
        base.update(cfg)  # the model config only adds `model:`
        cfg = base
    return cfg


def flatten(d, prefix=""):
    """{'train': {'lr': 1e-4}} -> {'train.lr': 1e-4}, for MLflow params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def forever(loader):
    """Endless batches: training is counted in steps, not epochs."""
    while True:
        yield from loader


def to_tensor(x):
    return x.as_tensor() if hasattr(x, "as_tensor") else x  # MONAI MetaTensor -> plain tensor


def warmup_cosine(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


# ---------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, loader, loss_fn, cfg, device, step):
    model.eval()
    crop = tuple(cfg["data"]["crop_size"])
    dice = DiceMetric(include_background=True, reduction="mean_batch")
    losses, pictures = [], []
    for i, batch in enumerate(loader):
        x = to_tensor(batch["image"]).to(device)
        y = to_tensor(batch["label"]).to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            logits = sliding_window_inference(x, crop, 4, model, overlap=cfg["train"]["val_overlap"])
        logits = logits.float()
        losses.append(loss_fn(logits, y).item())
        pred = (torch.sigmoid(logits) > 0.5).float()
        dice(y_pred=pred, y=y)
        if i < cfg["train"]["val_image_cases"]:
            pictures.append((batch["case_id"][0], x[0, 0].cpu(), y[0].cpu(), pred[0].cpu()))
    per_region = dice.aggregate().cpu().tolist()
    model.train()
    metrics = {f"val_dice_{r.lower()}": d for r, d in zip(REGIONS, per_region)}
    metrics["val_dice_mean"] = float(np.mean(per_region))
    metrics["val_loss"] = float(np.mean(losses))
    for case_id, flair, y, pred in pictures:
        mlflow.log_figure(picture(flair, y, pred, case_id, step), f"val_images/{case_id}/step_{step:06d}.png")
    plt.close("all")
    return metrics


def picture(flair, y, pred, case_id, step):
    """FLAIR | true regions | predicted regions, on the slice with the most tumour."""
    z = int(y[0].sum(dim=(0, 1)).argmax())
    def overlay(m):  # 1 = WT only, 2 = TC, 3 = ET (highest region wins)
        return (m[0] + m[1] + m[2])[:, :, z].T
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, title, over in zip(ax, ["FLAIR", "true", "predicted"], [None, overlay(y), overlay(pred)]):
        a.imshow(flair[:, :, z].T, cmap="gray", origin="lower")
        if over is not None:
            a.imshow(np.ma.masked_equal(over.numpy(), 0), cmap="autumn_r", alpha=0.6, origin="lower", vmin=1, vmax=3)
        a.set_title(title)
        a.axis("off")
    fig.suptitle(f"{case_id}  step {step}")
    return fig


def save_curves(history, path):
    """Training vs validation loss, and validation Dice per region, as one PNG."""
    tr = [h for h in history if "train_loss" in h]
    va = [h for h in history if "val_loss" in h]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([h["step"] for h in tr], [h["train_loss"] for h in tr], label="train loss")
    ax[0].plot([h["step"] for h in va], [h["val_loss"] for h in va], "o-", label="validation loss")
    ax[0].set_xlabel("step"); ax[0].set_title("Loss"); ax[0].legend()
    for key in ["val_dice_wt", "val_dice_tc", "val_dice_et", "val_dice_mean"]:
        ax[1].plot([h["step"] for h in va], [h[key] for h in va], "o-", label=key.replace("val_dice_", "").upper())
    ax[1].set_xlabel("step"); ax[1].set_ylim(0, 1); ax[1].set_title("Validation Dice"); ax[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- training
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, help="override (e.g. for a quick smoke test)")
    p.add_argument("--val-every", type=int, help="override")
    p.add_argument("--limit", type=int, help="use only the first N cases per split (smoke test)")
    p.add_argument("--run-name", help="MLflow run name (default: model name)")
    p.add_argument("--resume", help="path to a last.pt checkpoint to continue from")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.val_every:
        cfg["train"]["val_every"] = args.val_every
    tc, dc = cfg["train"], cfg["data"]
    name = cfg["model"]["name"]
    run_name = args.run_name or name

    seed_everything(cfg["seed"])
    device = torch.device("cuda")
    train_loader, val_loader = get_loaders(
        crop_size=tuple(dc["crop_size"]), batch_size=dc["batch_size"],
        num_workers=dc["num_workers"], cache_workers=dc["cache_workers"], limit=args.limit,
    )

    model = build_model(**cfg["model"]).to(device)
    loss_fn = DiceCELoss(sigmoid=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(tc["warmup_steps"], tc["max_steps"]))

    run_dir = Path("output/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_step, best = ck["step"], ck["best"]
        print(f"Resumed from step {start_step} (best mean Dice so far {best:.4f})")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["experiment"])
    try:
        mlflow.enable_system_metrics_logging()  # GPU / CPU / RAM use over time
    except Exception:
        pass

    history = []
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(cfg))
        mlflow.log_params({"git_commit": git_commit(), "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                           "train_cases": len(train_loader.dataset), "val_cases": len(val_loader.dataset)})
        mlflow.set_tag("model", name)

        batches = forever(train_loader)
        model.train()
        t0, running = time.perf_counter(), []
        for step in range(start_step + 1, tc["max_steps"] + 1):
            batch = next(batches)
            x = to_tensor(batch["image"]).to(device, non_blocking=True)
            y = to_tensor(batch["label"]).to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["amp"]):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            loss.backward()
            opt.step()
            sched.step()
            running.append(loss.item())

            if step % tc["log_every"] == 0:
                row = {"step": step, "train_loss": float(np.mean(running)), "lr": sched.get_last_lr()[0]}
                mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=step)
                history.append(row)
                rate = (time.perf_counter() - t0) / tc["log_every"]
                eta_h = (tc["max_steps"] - step) * rate / 3600
                print(f"step {step:6d}  loss {row['train_loss']:.4f}  lr {row['lr']:.2e}  {rate:.3f} s/step  ETA {eta_h:.1f} h", flush=True)
                t0, running = time.perf_counter(), []

            if step % tc["val_every"] == 0 or step == tc["max_steps"]:
                metrics = validate(model, val_loader, loss_fn, cfg, device, step)
                mlflow.log_metrics(metrics, step=step)
                history.append({"step": step, **metrics})
                save_curves(history, Path("results") / run_name / "curves.png")
                is_best = metrics["val_dice_mean"] > best
                if is_best:
                    best = metrics["val_dice_mean"]
                    torch.save({"model": model.state_dict(), "config": cfg, "step": step, "val_dice_mean": best}, run_dir / "best.pt")
                    mlflow.log_metric("best_val_dice_mean", best, step=step)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best, "config": cfg}, run_dir / "last.pt")
                print(f"  VAL step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                      + ("  << new best" if is_best else ""), flush=True)
                t0 = time.perf_counter()  # don't count validation time in s/step

        # end of training: keep the best checkpoint and the curves
        mlflow.log_artifact(str(run_dir / "best.pt"), artifact_path="checkpoints")
        out = Path("results") / run_name
        out.mkdir(parents=True, exist_ok=True)
        cols = ["step", "train_loss", "lr", "val_loss", "val_dice_wt", "val_dice_tc", "val_dice_et", "val_dice_mean"]
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(history)
        mlflow.log_artifact(str(out / "history.csv"))
        mlflow.log_artifact(str(out / "curves.png"))
        print(f"Done. Best mean validation Dice {best:.4f}. Curves: {out / 'history.csv'}")


if __name__ == "__main__":
    main()
