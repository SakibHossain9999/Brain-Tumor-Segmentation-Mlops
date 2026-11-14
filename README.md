# 3D Brain Tumor Segmentation: Model Benchmark, MLOps Pipeline & Interactive 3D Viewer

![CI](https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops/actions/workflows/ci.yml/badge.svg)

Three 3D segmentation networks trained and compared on multi-parametric brain MRI (FLAIR, T1, T1-contrast, T2),
served on CPU through ONNX Runtime behind a FastAPI upload service, with a three.js viewer that renders the brain
and tumour sub-regions in 3D, including non-photorealistic (toon / hatching) styles.

## Results (held-out test set, 72 brains, mean Dice over WT / TC / ET)

| Model | Test Dice | CPU time per brain (2 threads, ONNX) |
|---|---|---|
| **SegResNet** (champion) | **0.835** | 7.1 s |
| Swin UNETR | 0.835 | 31.3 s |
| SegFormer3D | 0.813 | 1.4 s |

SegResNet and Swin UNETR are statistically indistinguishable (Wilcoxon p = 0.42); both beat SegFormer3D (p < 0.0001).
SegResNet is the champion: same accuracy, about 4× faster on CPU than Swin UNETR. Details: [`results/comparison.md`](results/comparison.md).

## What's in here

| Folder | What it does |
|---|---|
| `ml/` | training (MONAI, PyTorch), evaluation (Dice, HD95, sensitivity, precision), ONNX export, MLflow model registry |
| `pipeline/brainviz/` | MRI + labels → 3D meshes (glTF) and compressed scan volumes for the viewer |
| `api/` | FastAPI service: upload scans (MSD 4-in-1 or BraTS 4-file), queue a job, run any of the 3 models, compare with an uploaded expert outline |
| `web/` | TypeScript + three.js viewer and a Models page (training curves, test results, accuracy vs speed) |
| `tests/` | unit tests on synthetic data (labels, meshing, upload checks, metrics, inference) |
| `Dockerfile`, `.github/workflows/ci.yml` | CI: tests → viewer build → Docker image built and tested end to end at 2 CPUs / 16 GB → pushed to GHCR |

Large files (ONNX models, sample brains, gallery cases) are stored on the Hugging Face Hub ([MohammadSakibHossain/brain-tumor-segmentation-mlops](https://huggingface.co/MohammadSakibHossain/brain-tumor-segmentation-mlops)) and fetched by `scripts/hub.py`.

## Run it

```bash
docker build -t brain-seg .
docker run -p 7860:7860 brain-seg        # http://localhost:7860
```

## Data and licence notes

Trained on the Medical Segmentation Decathlon Task01_BrainTumour (derived from BraTS 2016/2017), CC-BY-SA 4.0.
For research and education only: this is not a medical device.
