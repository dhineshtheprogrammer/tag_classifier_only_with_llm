from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def preprocess(
    image_path: str | Path,
    config: dict,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if img.shape[0] < 100 or img.shape[1] < 100:
        print(f"[preprocess] Warning: very small image {img.shape} — template matching may find nothing")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Auto-polarity: valve lines should be white (255) on black (0)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    ksize = config.get("preprocess", {}).get("median_ksize", 3)
    denoised = cv2.medianBlur(binary, ksize)

    if config.get("preprocess", {}).get("deskew", True):
        denoised = _deskew(denoised)

    if debug and debug_dir is not None:
        stem = Path(image_path).stem
        out = Path(debug_dir) / f"{stem}_preprocessed.png"
        cv2.imwrite(str(out), denoised)

    return denoised


def _estimate_skew(binary: np.ndarray) -> float:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )
    if lines is None:
        return 0.0
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    horiz = [a for a in angles if abs(a) < 45]
    return float(np.median(horiz)) if horiz else 0.0


def _deskew(binary: np.ndarray) -> np.ndarray:
    skew = _estimate_skew(binary)
    if abs(skew) < 0.5:
        return binary
    h, w = binary.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), skew, 1.0)
    return cv2.warpAffine(
        binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )
