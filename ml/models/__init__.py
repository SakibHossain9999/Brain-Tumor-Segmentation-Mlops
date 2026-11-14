"""One place to build every model, so training, evaluation and the API all use identical settings."""
from monai.networks.nets import SegResNet, SwinUNETR

from .segformer3d import SegFormer3D

MODEL_NAMES = ("segresnet", "swin_unetr", "segformer3d")


def build_model(name: str, in_channels: int = 4, out_channels: int = 3, **kwargs):
    """Return an untrained model: 4 MRI scans in, 3 region scores (WT, TC, ET) out."""
    if name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=kwargs.get("init_filters", 16),
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=kwargs.get("dropout", 0.2),
        )
    if name == "swin_unetr":
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kwargs.get("feature_size", 48),
            use_checkpoint=kwargs.get("use_checkpoint", False),
        )
    if name == "segformer3d":
        return SegFormer3D(in_channels=in_channels, num_classes=out_channels)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")