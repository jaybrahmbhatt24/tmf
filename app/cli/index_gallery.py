from __future__ import annotations

import argparse
import os
from typing import Iterable

import cv2

from app.db import GALLERY_DIR, get_connection, ensure_schema, insert_face
from app.face import detect_faces_bgr, crop_with_margin, compute_phash, read_image_bgr


def iter_image_files(root: str) -> Iterable[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name.lower())
            if ext in exts:
                yield os.path.join(dirpath, name)


def index_gallery(gallery_dir: str) -> None:
    conn = get_connection()
    ensure_schema(conn)

    images = list(iter_image_files(gallery_dir))
    for idx, path in enumerate(images, 1):
        try:
            img = read_image_bgr(path)
        except Exception:
            continue

        faces = detect_faces_bgr(img)
        if not faces:
            continue
        for box in faces:
            crop = crop_with_margin(img, box, margin_ratio=0.25)
            ph = compute_phash(crop)
            insert_face(conn, image_path=os.path.abspath(path), x=box.x, y=box.y, w=box.w, h=box.h, phash=ph)

        if idx % 50 == 0:
            print(f"Indexed {idx}/{len(images)} images...")

    print("Indexing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index faces from gallery directory")
    parser.add_argument("--gallery", default=GALLERY_DIR, help="Path to gallery directory")
    args = parser.parse_args()
    index_gallery(args.gallery)


if __name__ == "__main__":
    main()
