"""Stand-alone ConvNeXt-Small image encoder.

A plain-PyTorch re-implementation of the mmcls 1.x ``ConvNeXt`` backbone whose
parameter names match the checkpoints produced by
``configs/convnext_s_mmcls.py``. It lets the 768-d image embeddings be extracted
without installing the (pinned, release-candidate) mmcv/mmcls stack used for
training.

Embedding = global average pooling of the last stage followed by the final
LayerNorm (mmcls ``gap_before_final_norm=True``), i.e. exactly the input of the
linear classification head.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CONVNEXT_SMALL = dict(depths=(3, 3, 27, 3), channels=(96, 192, 384, 768))

# Intensity normalisation used when the backbone was trained (8-bit, BGR->RGB).
PIXEL_MEAN = 77.52425988
PIXEL_STD = 51.8555656
INPUT_SIZE = 1280  # images are resized (keep ratio, bicubic) and padded to 1280x1280


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(self.depthwise_conv(x))
        x = x.permute(0, 2, 3, 1)
        x = self.pointwise_conv2(self.act(self.pointwise_conv1(x)))
        x = x.permute(0, 3, 1, 2) * self.gamma.view(1, -1, 1, 1)
        return shortcut + x  # drop-path is identity at inference


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt backbone returning one 768-d embedding per image."""

    def __init__(self, depths=CONVNEXT_SMALL["depths"], channels=CONVNEXT_SMALL["channels"]):
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, channels[0], 4, stride=4), LayerNorm2d(channels[0], eps=1e-6))]
        )
        for i in range(1, len(channels)):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(channels[i - 1], eps=1e-6),
                    nn.Conv2d(channels[i - 1], channels[i], 2, stride=2),
                )
            )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[ConvNeXtBlock(c) for _ in range(d)]) for d, c in zip(depths, channels)]
        )
        self.norm3 = nn.LayerNorm(channels[-1], eps=1e-6)
        self.embed_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
        return self.norm3(x.mean(dim=(-2, -1)))

    @classmethod
    def from_mmcls_checkpoint(cls, path: str | Path, map_location="cpu") -> "ConvNeXtEncoder":
        """Load the backbone weights from an mmcls/mmengine checkpoint.

        Uses ``weights_only=True`` so no code stored in the pickle is executed.
        Auxiliary heads (cancer, view, BI-RADS, density, difficulty) are ignored.
        """
        ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        model = cls()
        model.load_state_dict(backbone, strict=True)
        return model.eval()


def preprocess_for_convnext(image: "np.ndarray", size: int = INPUT_SIZE) -> torch.Tensor:  # noqa: F821
    """Replicates the mmcls test pipeline: keep-ratio bicubic resize, zero pad, normalise.

    ``image`` is an 8-bit grayscale (H, W) or BGR (H, W, 3) array as read by ``cv2.imread``.
    Returns a (3, size, size) float tensor.
    """
    import cv2
    import numpy as np

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = image[..., ::-1]  # BGR -> RGB (no-op for grayscale replicated to 3 channels)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = image
    # mmcls pads with raw intensity 0 *before* normalisation (Pad runs before the
    # data preprocessor), so padded pixels end up at -mean/std, as here.
    canvas = (canvas - PIXEL_MEAN) / PIXEL_STD
    return torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
