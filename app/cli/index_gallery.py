from __future__ import annotations

import argparse
import os
from typing import Iterable

import cv2

from app.db import (
    GALLERY_DIR,
    get_connection,
    ensure_schema,
    insert_face,
    create_or_get_event,
    upsert_asset,
    insert_asset_face,
    relpath_from_gallery,
)
from app.face import detect_faces_bgr, crop_with_margin, compute_phash, read_image_bgr


def iter_image_files(root: str) -> Iterable[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name.lower())
            if ext in exts:
                yield os.path.join(dirpath, name)


def index_gallery(gallery_dir: str, event_name: str) -> None:
    conn = get_connection()
    ensure_schema(conn)
    event_id = create_or_get_event(conn, name=event_name)

    images = list(iter_image_files(gallery_dir))
    for idx, path in enumerate(images, 1):
        try:
            img = read_image_bgr(path)
        except Exception:
            continue

        faces = detect_faces_bgr(img)
        if not faces:
            continue
        # Register asset and faces for new schema
        rel_path = relpath_from_gallery(os.path.abspath(path))
        asset_id = upsert_asset(conn, event_id=event_id, rel_path=rel_path)
        for box in faces:
            crop = crop_with_margin(img, box, margin_ratio=0.25)
            ph = compute_phash(crop)
            insert_asset_face(conn, asset_id=asset_id, x=box.x, y=box.y, w=box.w, h=box.h, phash=ph)
            # Also keep legacy table to support /search endpoint
            insert_face(conn, image_path=os.path.abspath(path), x=box.x, y=box.y, w=box.w, h=box.h, phash=ph)

        if idx % 50 == 0:
            print(f"Indexed {idx}/{len(images)} images...")

    print("Indexing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index faces from gallery directory")
    parser.add_argument("--gallery", default=GALLERY_DIR, help="Path to gallery directory")
    parser.add_argument("--event", default="default", help="Event name for indexing")
    args = parser.parse_args()
    index_gallery(args.gallery, args.event)


if __name__ == "__main__":
    main()
