| Model | Dice WT | Dice TC | Dice ET | **Dice mean** | HD95 WT | HD95 TC | HD95 ET | Sens. ET | Prec. ET | Params (M) | GFLOPs / window | Predict (s/case) | GPU (GB) | Train (s/step) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SegFormer3D | 0.889 | 0.815 | 0.736 | **0.813** | 8.0 | 8.6 | 10.2 | 0.788 | 0.733 | 4.53 | 71.4 | 0.108 | 1.91 | 0.097 |
| SegResNet | 0.899 | 0.833 | 0.772 | **0.835** | 10.0 | 9.2 | 16.7 | 0.808 | 0.790 | 4.7 | 578.2 | 0.217 | 2.95 | 0.334 |
| Swin UNETR | 0.898 | 0.829 | 0.779 | **0.835** | 8.7 | 8.5 | 16.4 | 0.801 | 0.820 | 62.19 | 2978.5 | 0.554 | 12.84 | 1.659 |

Dice: overlap with the expert outline (1 = perfect). HD95: distance between predicted and true tumour edges, mm (lower is better; a missed or invented region counts as 373 mm). Means over cases.
