"""One prediction path, shared by test evaluation and the API, so both give identical results.

PyTorch checkpoint (GPU, used for evaluation):
    model, cfg, ckpt = load_model("output/runs/segresnet/best.pt", "cuda")
ONNX file (CPU, used by the API; no model code or GPU needed):
    model, cfg = load_onnx("models/segresnet.onnx")
Then, either way:
    probs = predict(model, image, cfg)          # image: (4, H, W, D) raw scans -> (3, H, W, D) WT/TC/ET probabilities
    labels = regions_to_labels(probs > 0.5)     # -> (H, W, D) 0-3, same coding as the MSD label files
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# the same normalisation as training (ml/data.py): brain voxels of each scan -> mean 0, std 1
_normalise = NormalizeIntensity(nonzero=True, channel_wise=True)


def load_model(ckpt_path, device="cuda"):
    """A trained checkpoint -> (model ready to predict, its training config, the checkpoint dict)."""
    from ml.models import build_model  # only needed for PyTorch checkpoints, not for ONNX

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file: it also holds the config
    cfg = ckpt["config"]
    model = build_model(**cfg["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt


class OnnxModel:
    """An ONNX file that behaves like a PyTorch model for predict(): tensor in, tensor out (on the CPU).

    The file takes exactly one 160^3 window (batch 1), so a batch of windows is run one by one.
    """

    def __init__(self, path, threads=None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads  # CPU cores to use; None = all of them
        # onnxruntime normally keeps its working memory for re-use. Measured on one brain (2 threads):
        # Swin UNETR peaks at 20.7 GB with that cache and 12.0 GB without, for ~3 s more. Off, so the
        # service fits a 16 GB machine.
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        x = x.detach().float().cpu().numpy()
        outs = [self.session.run(None, {self.input_name: x[i : i + 1]})[0] for i in range(len(x))]
        return torch.from_numpy(np.concatenate(outs))


def load_onnx(path, threads=None):
    """models/<name>.onnx + its models/<name>.json (training config, written at export) -> (model, cfg)."""
    path = Path(path)
    cfg = json.loads(path.with_suffix(".json").read_text())["config"]
    return OnnxModel(path, threads), cfg


@torch.no_grad()
def predict(model, image, cfg, overlap=0.5, sw_batch_size=4):
    """Raw scans (4, H, W, D) -> region probabilities (3, H, W, D), on the model's device.

    Same steps as validation during training:
      1. crop to the box around the brain (the black border carries nothing), so one 160-wide
         window spans the brain side to side and window edges don't cut through the middle of it
      2. normalise each scan (brain voxels -> mean 0, std 1)
      3. cover the crop with overlapping 160^3 windows (sliding window) and blend their predictions;
         a crop smaller than a window is padded automatically
      4. put the result back into a full-size, zero-filled volume
    """
    device = next(model.parameters()).device if isinstance(model, torch.nn.Module) else torch.device("cpu")
    image = torch.as_tensor(image, dtype=torch.float32)
    idx = torch.nonzero((image > 0).any(0))  # brain voxels (same rule as CropForegroundd)
    box = tuple(slice(int(a), int(b) + 1) for a, b in zip(idx.min(0).values, idx.max(0).values))
    x = torch.as_tensor(_normalise(image[(slice(None), *box)]))[None].to(device)  # (1, 4, h, w, d)
    crop = tuple(cfg["data"]["crop_size"])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"] and device.type == "cuda"):
        logits = sliding_window_inference(x, crop, sw_batch_size, model, overlap=overlap)
    probs = torch.zeros((logits.shape[1], *image.shape[1:]), device=device)
    probs[(slice(None), *box)] = torch.sigmoid(logits.float())[0]
    return probs


def regions_to_labels(regions):
    """(3, H, W, D) WT/TC/ET masks -> (H, W, D) label map: 1 edema, 2 non-enhancing, 3 enhancing (MSD coding)."""
    regions = torch.as_tensor(regions).bool()
    labels = torch.zeros(regions.shape[1:], dtype=torch.uint8, device=regions.device)
    labels[regions[0]] = 1  # inner regions overwrite outer ones
    labels[regions[1]] = 2
    labels[regions[2]] = 3
    return labels.cpu().numpy().astype(np.uint8)
