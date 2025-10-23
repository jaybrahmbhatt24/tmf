from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for frontal face detection.")
    return detector


_FACE_DETECTOR: cv2.CascadeClassifier | None = None


def detect_faces_bgr(image_bgr: np.ndarray) -> List[FaceBox]:
    """Detect faces in a BGR image using Haar cascades.

    Returns faces sorted by area (largest first).
    """
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = _get_face_detector()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detections = _FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(60, 60),
    )

    faces = [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]
    faces.sort(key=lambda b: b.w * b.h, reverse=True)
    return faces


def crop_with_margin(image_bgr: np.ndarray, box: FaceBox, margin_ratio: float = 0.2) -> np.ndarray:
    """Crop a face region with margin; clamps to image bounds."""
    h, w = image_bgr.shape[:2]
    mx = int(box.w * margin_ratio)
    my = int(box.h * margin_ratio)
    x1 = max(0, box.x - mx)
    y1 = max(0, box.y - my)
    x2 = min(w, box.x + box.w + mx)
    y2 = min(h, box.y + box.h + my)
    return image_bgr[y1:y2, x1:x2]


def compute_phash(image_bgr: np.ndarray) -> int:
    """Compute a 64-bit perceptual hash (pHash) using DCT.

    Steps:
    - Convert to grayscale
    - Resize to 32x32
    - Compute 2D DCT (cv2.dct)
    - Take top-left 8x8 coefficients (excluding [0,0])
    - Threshold by median to form 64-bit hash
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8]

    # Exclude DC component at (0,0) for median computation
    dct_flat = dct_low.flatten()
    dct_sub = dct_flat[1:]
    median_val = np.median(dct_sub)

    bits = (dct_low > median_val).astype(np.uint8).flatten()

    # Build 64-bit integer from bits[0..63]
    h_val = 0
    for bit in bits:
        h_val = (h_val << 1) | int(bit)
    return int(h_val)


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def read_image_bgr(path: str) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        # Fallback for environments without fromfile support on path
        data = cv2.imread(path, cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return data
