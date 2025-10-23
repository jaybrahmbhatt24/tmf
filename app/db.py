from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .face import hamming_distance

DEFAULT_DB_PATH = os.environ.get("TMF_DB_PATH", os.path.join("data", "index.db"))
GALLERY_DIR = os.environ.get("TMF_GALLERY_DIR", os.path.join("data", "gallery"))


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    phash INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_path, x, y, w, h, phash) ON CONFLICT IGNORE
);
CREATE INDEX IF NOT EXISTS idx_faces_image_path ON faces(image_path);
"""


@dataclass(frozen=True)
class FaceRecord:
    id: int
    image_path: str
    x: int
    y: int
    w: int
    h: int
    phash: int


@dataclass(frozen=True)
class RankedImage:
    image_path: str
    best_distance: int


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_face(conn: sqlite3.Connection, image_path: str, x: int, y: int, w: int, h: int, phash: int) -> int | None:
    cur = conn.execute(
        "INSERT OR IGNORE INTO faces(image_path, x, y, w, h, phash) VALUES (?, ?, ?, ?, ?, ?)",
        (image_path, int(x), int(y), int(w), int(h), int(phash)),
    )
    conn.commit()
    return cur.lastrowid if cur.lastrowid != 0 else None


def iter_faces(conn: sqlite3.Connection) -> Iterable[FaceRecord]:
    for row in conn.execute("SELECT id, image_path, x, y, w, h, phash FROM faces"):
        yield FaceRecord(
            id=row["id"],
            image_path=row["image_path"],
            x=row["x"],
            y=row["y"],
            w=row["w"],
            h=row["h"],
            phash=row["phash"],
        )


def rank_by_phash(conn: sqlite3.Connection, query_phash: int, max_results: int = 100, distance_threshold: int | None = None) -> List[RankedImage]:
    """Return best image-level matches by Hamming distance across faces.

    Deduplicates by image_path using the best (minimum) distance among its faces.
    """
    best_by_image: dict[str, int] = {}
    for row in conn.execute("SELECT image_path, phash FROM faces"):
        img = row["image_path"]
        dist = hamming_distance(query_phash, int(row["phash"]))
        if distance_threshold is not None and dist > distance_threshold:
            continue
        prev = best_by_image.get(img)
        if prev is None or dist < prev:
            best_by_image[img] = dist

    ranked = [RankedImage(image_path=path, best_distance=dist) for path, dist in best_by_image.items()]
    ranked.sort(key=lambda r: r.best_distance)
    return ranked[:max_results]


def relpath_from_gallery(abs_path: str) -> str:
    base = os.path.abspath(GALLERY_DIR)
    return os.path.relpath(os.path.abspath(abs_path), start=base)


def abspath_in_gallery(rel_path: str) -> str:
    # Prevent path traversal
    base = os.path.abspath(GALLERY_DIR)
    safe_path = os.path.abspath(os.path.normpath(os.path.join(base, rel_path)))
    if os.path.commonpath([safe_path, base]) != base:
        raise ValueError("Attempted path outside gallery directory")
    return safe_path
