"""Extract one 768-d ConvNeXt embedding per mammogram.

Input: a CSV with columns year, patient_id, image_id, laterality (L/R), view (CC/MLO)
and path (DICOM or already-preprocessed TIFF/PNG). DICOMs are preprocessed on the fly
with ``mammo_report.preprocessing``; TIFF/PNG files are assumed to be preprocessed
already (as the ``*Anon_Transformed`` TIFFs used in the paper).

Output: the same rows plus Column1..Column768 (the format read by ``mammo_report.data``).

Example:
    python scripts/extract_features.py --images images.csv --checkpoint trained_model.pth --out features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mammo_report.convnext import ConvNeXtEncoder, preprocess_for_convnext  # noqa: E402
from mammo_report.preprocessing import load_average_cdf, preprocess_dicom, to_uint8  # noqa: E402


def load_image(path: str, cdf: np.ndarray, use_dicom_spacing: bool) -> np.ndarray:
    import cv2

    if path.lower().endswith(".dcm"):
        return to_uint8(preprocess_dicom(path, cdf, use_dicom_spacing).image)
    img = cv2.imread(path)  # 8-bit BGR, as in the original mmcls pipeline
    if img is None:
        raise FileNotFoundError(path)
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--checkpoint", required=True, help="ConvNeXt-S mmcls checkpoint (trained_model.pth)")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use-dicom-spacing", action="store_true",
                   help="resample with the DICOM PixelSpacing (published features used a fixed 0.065 mm)")
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNeXtEncoder.from_mmcls_checkpoint(a.checkpoint).to(device)
    cdf = load_average_cdf()
    df = pd.read_csv(a.images)
    feats = []
    for start in range(0, len(df), a.batch_size):
        rows = df.iloc[start:start + a.batch_size]
        x = torch.stack([preprocess_for_convnext(load_image(pth, cdf, a.use_dicom_spacing)) for pth in rows["path"]])
        with torch.no_grad(), torch.autocast(device, enabled=a.fp16):
            feats.append(model(x.to(device)).float().cpu().numpy())
        print(f"{min(start + a.batch_size, len(df))}/{len(df)}", end="\r")
    emb = pd.DataFrame(np.concatenate(feats), columns=[f"Column{i}" for i in range(1, 769)])
    pd.concat([df.reset_index(drop=True), emb], axis=1).to_csv(a.out, index=False)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
