---
license: cc-by-sa-4.0
library_name: onnx
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - brain-tumor-segmentation
  - 3d-segmentation
  - mri
  - brats
  - onnx
  - monai
  - segresnet
  - swin-unetr
  - segformer3d
datasets:
  - Medical Segmentation Decathlon (Task01 BrainTumour)
metrics:
  - dice
---

# 3D Brain Tumor Segmentation: ONNX models + demo data

Three 3D networks that outline a brain tumour in multi-parametric MRI, exported to **ONNX** for fast CPU inference,
plus the sample brains and 3D gallery used by the project's web app.

**Code, training, evaluation, API, viewer and CI/CD:**
[github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops](https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops)

> **Not a medical device.** For research and education only. Do not use for diagnosis or treatment decisions.

## What's in this repository

| Path | Contents | Size |
|---|---|---|
| `models/segresnet.onnx` | **SegResNet: the champion** (best accuracy for its cost) | 19 MB |
| `models/swin_unetr.onnx` | Swin UNETR: transformer, same accuracy, much slower | 249 MB |
| `models/segformer3d.onnx` | SegFormer3D: lightweight transformer, fastest, slightly less accurate | 18 MB |
| `models/<model>.json` | input/output spec, full training config, best checkpoint step, validation/test Dice, ONNX-vs-PyTorch agreement, CPU timings | |
| `models/manifest.json` | registry snapshot from MLflow: champion, reason, versions, SHA-256 checksums | |
| `samples/` | 4 test-set brains (`BRATS_075`, `095`, `156`, `238`): all 4 MRI scans in one file + the expert outline (`*_seg.nii.gz`) | 44 MB |
| `cases/` | the web viewer's gallery: 6 test brains as 3D meshes (glTF) + compressed scan volumes + `meta.json` | 99 MB |

## The task in plain words

Each patient has 4 MRI scans of the same head (FLAIR, T1, T1 with contrast, T2), each a 3D grid of
240 × 240 × 155 voxels at 1 mm, skull already removed. The tumour has three nested regions:

| Output channel | Region | Meaning |
|---|---|---|
| 0 | **WT**, whole tumour | swelling + inactive tumour + active tumour |
| 1 | **TC**, tumour core | inactive + active tumour (what a surgeon removes) |
| 2 | **ET**, enhancing tumour | active, growing tumour |

## Model specification

| | |
|---|---|
| Input | `scans`, float32, shape `[1, 4, 160, 160, 160]`, channels **FLAIR, T1, T1 + contrast, T2** |
| Output | `logits`, float32, shape `[1, 3, 160, 160, 160]`, channels **WT, TC, ET** |
| Post-processing | `sigmoid(logits) > 0.5`, independently per channel (a voxel can be in several regions) |
| Preprocessing | per scan, normalise the non-zero (brain) voxels to mean 0, std 1; crop to the brain's bounding box |
| Whole brains | slide a 160³ window over the cropped brain with 50% overlap, average the overlapping logits |
| Export | PyTorch 2.11 dynamo exporter, opset 18; the graph takes exactly one window |

## Results

Held-out test set of **72 brains**, never used for training or model selection.

| Model | Dice WT | Dice TC | Dice ET | **Mean Dice** | HD95 WT / TC / ET (mm) | Params | CPU s / brain (2 / 8 threads) | ONNX vs PyTorch |
|---|---|---|---|---|---|---|---|---|
| **SegResNet** | 0.899 | 0.833 | 0.772 | **0.835** | 10.0 / 9.2 / 16.7 | 4.7 M | 7.1 / 3.0 | 0.998 |
| Swin UNETR | 0.898 | 0.829 | 0.779 | **0.835** | 8.7 / 8.5 / 16.4 | 62.2 M | 31.3 / 14.2 | 0.994 |
| SegFormer3D | 0.889 | 0.815 | 0.736 | **0.813** | 8.0 / 8.6 / 10.2 | 4.5 M | 1.4 / 0.8 | 0.983 |

- **Dice:** overlap with the expert outline (1 = identical). **HD95:** 95th-percentile distance between predicted and
  expert edges; a missed or invented region counts as 373 mm. **ONNX vs PyTorch:** Dice between the two versions' masks.
- **Statistics** (Wilcoxon signed-rank on per-brain mean Dice): SegResNet vs Swin UNETR p = 0.42 (no difference);
  both vs SegFormer3D p < 0.0001.
- **Champion: SegResNet.** It ties Swin UNETR with 13× fewer parameters, 4× faster CPU inference and 4× less GPU
  memory.

## How to use

```python
import numpy as np, nibabel as nib, onnxruntime as ort, torch
from huggingface_hub import hf_hub_download
from monai.inferers import sliding_window_inference

repo = "MohammadSakibHossain/brain-tumor-segmentation-mlops"
session = ort.InferenceSession(hf_hub_download(repo, "models/segresnet.onnx"))
scan_file = hf_hub_download(repo, "samples/BRATS_075.nii.gz")   # or your own: 4 scans, skull-stripped, 1 mm

x = np.moveaxis(nib.load(scan_file).get_fdata(dtype=np.float32), -1, 0)   # (4, H, W, D): FLAIR, T1, T1c, T2
for c in range(4):                                       # normalise each scan's brain voxels
    v = x[c] != 0
    x[c][v] = (x[c][v] - x[c][v].mean()) / x[c][v].std()

brain = (x != 0).any(0)                                  # crop to the brain's bounding box
idx = np.nonzero(brain)
box = tuple(slice(i.min(), i.max() + 1) for i in idx)
crop = torch.from_numpy(x[(slice(None), *box)].copy())[None]

def onnx_model(windows):                                 # the ONNX graph takes one 160³ window at a time
    return torch.cat([torch.from_numpy(session.run(None, {"scans": w[None].numpy()})[0]) for w in windows])

logits = sliding_window_inference(crop, (160, 160, 160), sw_batch_size=1, predictor=onnx_model, overlap=0.5)
regions = np.zeros((3, *brain.shape), bool)              # back to full size: WT, TC, ET
regions[(slice(None), *box)] = (torch.sigmoid(logits[0]) > 0.5).numpy()

labels = np.zeros(brain.shape, np.uint8)                 # as one label map: 1 swelling, 2 inactive, 3 active
labels[regions[0]] = 1; labels[regions[1]] = 2; labels[regions[2]] = 3
```

Or run the whole app (upload page + 3D viewer) with Docker, see the GitHub README.

## Training

| | |
|---|---|
| Data | Medical Segmentation Decathlon Task01 BrainTumour, 484 labelled brains from BraTS 2016/2017 |
| Split | seed 42: 340 training / 72 validation / 72 test |
| Recipe (identical for all three) | random 160³ crops (half centred on tumour), batch 2; random flips on 3 axes, ±10% contrast and brightness; Dice + cross-entropy loss; AdamW, lr 1e-4, weight decay 1e-5; 500-step warm-up then cosine decay; 30,000 steps; bfloat16 mixed precision |
| Model selection | best validation Dice, checked every 1,000 steps (SegResNet step 15,000, Swin UNETR 20,000, SegFormer3D 24,000) |
| Hardware | one NVIDIA RTX A6000: SegFormer3D 0.8 h, SegResNet 2.8 h, Swin UNETR 13.8 h |
| Tracking | MLflow (runs, metrics, curves, registered model `brain-tumour-seg` with a `@champion` alias) |

## Limitations

- Trained on pre-operative gliomas from 2016/2017 data; other scanners, post-operative scans or tumour types may do worse.
- Inputs must be skull-stripped, roughly 1 mm, co-registered, and in the channel order above.
- Scores on training brains are optimistic: the four `samples/` and six `cases/` brains are all from the **test** set.

## Licences and citations

- **Data and derived files** (`samples/`, `cases/`, and the models trained on it): Medical Segmentation Decathlon,
  [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Simpson et al., *A large annotated medical image
  dataset for the development and evaluation of segmentation algorithms*, arXiv:1902.09063, 2019; derived from
  BraTS (Menze et al., IEEE TMI 2015; Bakas et al., Scientific Data 2017).
- **SegResNet:** Myronenko, *3D MRI brain tumor segmentation using autoencoder regularization*, BrainLes 2018.
- **Swin UNETR:** Hatamizadeh et al., *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in
  MRI Images*, BrainLes 2021.
- **SegFormer3D:** Perera, Navard, Yilmaz, *SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation*,
  CVPR Workshops 2024. Architecture code from [OSUPCVLab/SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D) (GPL-3.0).
