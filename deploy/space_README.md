---
title: 3D Brain Tumor Segmentation
emoji: 🧠
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: cc-by-sa-4.0
short_description: Upload brain MRI, get the tumour in 3D (3 models, ONNX, CPU)
models:
  - MohammadSakibHossain/brain-tumor-segmentation-mlops
---

# 3D Brain Tumor Segmentation: live demo

Upload four brain MRI scans (or use one of the sample brains), pick SegResNet, Swin UNETR or SegFormer3D,
and see the tumour's three regions in an interactive 3D viewer, compared with the expert's outline.

This Space is deployed automatically by GitHub Actions after the tests and the container's end-to-end test pass.
Code, training, evaluation and documentation:
[github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops](https://github.com/SakibHossain9999/Brain-Tumor-Segmentation-Mlops).
Models and sample data: [MohammadSakibHossain/brain-tumor-segmentation-mlops](https://huggingface.co/MohammadSakibHossain/brain-tumor-segmentation-mlops).

Free CPU hardware: the first visit after a quiet spell takes 1–2 minutes to wake up; SegResNet takes ~7 s per brain.

**Not a medical device.** Research and education only.
