"""Mammogram preprocessing (paper Sec. II-B).

Steps: inversion of negative images -> horizontal flip so the breast faces the
same side -> min-max normalisation -> resampling to 0.1 mm/pixel -> breast
segmentation -> histogram matching of the breast region to an average CDF
computed from 100 high-contrast RSNA mammograms (``assets/average_cdf.npy``).

Port of ``Code/Bety/Feature Extraction/preprocessing.py`` (Tec BCR lab repo,
revision f5c6bdf, Jan 2025).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SPACING_MM = 0.1
LEGACY_SPACING_MM = 0.065
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def load_average_cdf(path: str | Path = ASSETS / "average_cdf.npy") -> np.ndarray:
    return np.load(path)


@dataclass
class PreprocessedImage:
    image: np.ndarray        # float64 in [0, 1], background = 0
    breast_mask: np.ndarray  # uint8 {0, 1}
    was_inverted: bool
    was_flipped: bool
    pixel_spacing_mm: float


def histogram_matching(image_u16: np.ndarray, reference_cdf: np.ndarray) -> np.ndarray:
    """Map a 16-bit image so its CDF follows ``reference_cdf`` (65536 bins)."""
    from skimage.exposure import cumulative_distribution

    cdf, bins = cumulative_distribution(image_u16)
    cdf = np.insert(cdf, 0, [0] * bins[0])
    cdf = np.append(cdf, [1] * (65535 - bins[-1]))
    new_pixels = np.interp(cdf, reference_cdf, np.arange(65536))
    out = new_pixels[image_u16.ravel()].reshape(image_u16.shape)
    return (out - out.min()) / (out.max() - out.min())


def breast_mask(image_u16: np.ndarray, spacing_mm: float = TARGET_SPACING_MM) -> np.ndarray:
    """Threshold + morphological cleaning; returns the full breast mask (uint8)."""
    import cv2
    from skimage import morphology

    counts, edges = np.histogram(image_u16, bins=50)
    t1, cdf, target = 0, counts[0], counts.sum() * 0.3
    while cdf < target:
        t1 += 1
        cdf += counts[t1]
    mode = int(np.argmax(counts))
    threshold = (edges[mode + 1] + edges[t1 + 1]) / 2
    mask = (image_u16 > threshold).astype(np.uint8)
    large_disk = morphology.disk(round(5 / spacing_mm))  # 5 mm
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morphology.disk(10))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, large_disk)


def preprocess_array(
    pixels: np.ndarray,
    pixel_spacing_mm: float | None,
    reference_cdf: np.ndarray,
) -> PreprocessedImage:
    """Preprocess a raw mammogram pixel array.

    ``pixel_spacing_mm=None`` reproduces the published pipeline, which always
    used 0.065 mm because the DICOM spacing was never parsed successfully.
    """
    import cv2

    f = pixels.astype(np.float64)
    was_inverted = f.mean() > 2500  # MONOCHROME1-like images
    if was_inverted:
        f = f.max() - f
    w = f.shape[1]
    was_flipped = f[:, : w // 2].mean() < f[:, w // 2 :].mean()
    if was_flipped:
        f = np.fliplr(f)
    f = (f - f.min()) / (f.max() - f.min())

    spacing = LEGACY_SPACING_MM if pixel_spacing_mm is None else float(pixel_spacing_mm)
    h, w = f.shape
    new_size = (int(w * spacing / TARGET_SPACING_MM), int(h * spacing / TARGET_SPACING_MM))
    f = cv2.resize(f, new_size)
    f16 = (f * 65535).astype(np.uint16)

    mask = breast_mask(f16)
    matched = histogram_matching((f16 * mask).astype(np.uint16), reference_cdf)
    return PreprocessedImage(matched, mask, bool(was_inverted), bool(was_flipped), spacing)


def preprocess_dicom(path: str | Path, reference_cdf: np.ndarray, use_dicom_spacing: bool = False) -> PreprocessedImage:
    import pydicom

    ds = pydicom.dcmread(str(path))
    spacing = None
    if use_dicom_spacing and "PixelSpacing" in ds:
        spacing = float(ds.PixelSpacing[0])
    return preprocess_array(ds.pixel_array, spacing, reference_cdf)


def to_uint8(image01: np.ndarray) -> np.ndarray:
    """8-bit version fed to the ConvNeXt (what ``cv2.imread`` returns for the saved TIFFs)."""
    return np.clip(image01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_tiff(image01: np.ndarray, path: str | Path) -> None:
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (image01 * 65535).astype(np.uint16))
