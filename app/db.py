from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import secrets
import datetime as _dt

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

-- Users
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    contact TEXT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('photographer','customer','guest')),
    embedding_phash INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    owner_user_id INTEGER,
    couple_user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name) ON CONFLICT IGNORE,
    FOREIGN KEY(owner_user_id) REFERENCES users(id),
    FOREIGN KEY(couple_user_id) REFERENCES users(id)
);

-- Assets (photos/videos)
CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK(media_type IN ('image','video')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, path) ON CONFLICT IGNORE,
    FOREIGN KEY(event_id) REFERENCES events(id)
);
CREATE INDEX IF NOT EXISTS idx_assets_event ON assets(event_id);

-- Faces detected per asset
CREATE TABLE IF NOT EXISTS asset_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    phash INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, x, y, w, h, phash) ON CONFLICT IGNORE,
    FOREIGN KEY(asset_id) REFERENCES assets(id)
);
CREATE INDEX IF NOT EXISTS idx_asset_faces_asset ON asset_faces(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_faces_phash ON asset_faces(phash);

-- Guest access links
CREATE TABLE IF NOT EXISTS guest_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    created_by_user_id INTEGER,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(event_id) REFERENCES events(id),
    FOREIGN KEY(created_by_user_id) REFERENCES users(id)
);

-- Activity logs
CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Download logs
CREATE TABLE IF NOT EXISTS downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    asset_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
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


# ------------------------------- Users ------------------------------------

def create_user(
    conn: sqlite3.Connection,
    *,
    name: str,
    age: Optional[int],
    gender: Optional[str],
    contact: Optional[str],
    email: str,
    password_hash: str,
    role: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO users(name, age, gender, contact, email, password_hash, role)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (name, age, gender, contact, email.lower(), password_hash, role),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_email(conn: sqlite3.Connection, email: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
    row = cur.fetchone()
    return row


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
    return cur.fetchone()


def set_user_embedding_phash(conn: sqlite3.Connection, user_id: int, phash: int) -> None:
    conn.execute("UPDATE users SET embedding_phash = ? WHERE id = ?", (int(phash), int(user_id)))
    conn.commit()


# ------------------------------- Events/Assets -----------------------------

def create_or_get_event(conn: sqlite3.Connection, *, name: str, owner_user_id: Optional[int] = None, couple_user_id: Optional[int] = None) -> int:
    cur = conn.execute("INSERT OR IGNORE INTO events(name, owner_user_id, couple_user_id) VALUES(?, ?, ?)", (name, owner_user_id, couple_user_id))
    if cur.lastrowid:
        conn.commit()
        return int(cur.lastrowid)
    # fetch existing
    cur2 = conn.execute("SELECT id FROM events WHERE name = ?", (name,))
    row = cur2.fetchone()
    if not row:
        raise RuntimeError("Failed to create or fetch event")
    return int(row["id"])


def get_event_by_name(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM events WHERE name = ?", (name,))
    return cur.fetchone()


def get_event_by_id(conn: sqlite3.Connection, event_id: int) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM events WHERE id = ?", (int(event_id),))
    return cur.fetchone()


def upsert_asset(conn: sqlite3.Connection, *, event_id: int, rel_path: str, media_type: str = "image") -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO assets(event_id, path, media_type) VALUES(?, ?, ?)",
        (int(event_id), rel_path, media_type),
    )
    if cur.lastrowid:
        conn.commit()
        return int(cur.lastrowid)
    cur2 = conn.execute("SELECT id FROM assets WHERE event_id = ? AND path = ?", (int(event_id), rel_path))
    row = cur2.fetchone()
    if not row:
        raise RuntimeError("Failed to upsert asset")
    return int(row["id"])


def insert_asset_face(conn: sqlite3.Connection, *, asset_id: int, x: int, y: int, w: int, h: int, phash: int) -> Optional[int]:
    cur = conn.execute(
        "INSERT OR IGNORE INTO asset_faces(asset_id, x, y, w, h, phash) VALUES (?, ?, ?, ?, ?, ?)",
        (int(asset_id), int(x), int(y), int(w), int(h), int(phash)),
    )
    conn.commit()
    return int(cur.lastrowid) if cur.lastrowid != 0 else None


def list_assets_for_event(conn: sqlite3.Connection, event_id: int) -> List[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM assets WHERE event_id = ? ORDER BY id DESC", (int(event_id),))
    return list(cur.fetchall())


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


def rank_assets_by_phash(conn: sqlite3.Connection, query_phash: int, max_results: int = 100, distance_threshold: Optional[int] = None) -> List[RankedImage]:
    """Rank assets (photos) by best face distance."""
    best_by_asset: dict[int, int] = {}
    for row in conn.execute("SELECT asset_id, phash FROM asset_faces"):
        aid = int(row["asset_id"])
        dist = hamming_distance(query_phash, int(row["phash"]))
        if distance_threshold is not None and dist > distance_threshold:
            continue
        prev = best_by_asset.get(aid)
        if prev is None or dist < prev:
            best_by_asset[aid] = dist
    if not best_by_asset:
        return []
    ids = list(best_by_asset.keys())
    placeholders = ",".join(["?"] * len(ids))
    rows = list(conn.execute(f"SELECT id, path FROM assets WHERE id IN ({placeholders})", ids))
    path_by_id = {int(r["id"]): r["path"] for r in rows}
    ranked = [RankedImage(image_path=path_by_id[aid], best_distance=dist) for aid, dist in best_by_asset.items() if aid in path_by_id]
    ranked.sort(key=lambda r: r.best_distance)
    return ranked[:max_results]


def rank_assets_by_phash_for_event(
    conn: sqlite3.Connection,
    query_phash: int,
    event_id: int,
    max_results: int = 100,
    distance_threshold: Optional[int] = None,
) -> List[RankedImage]:
    best_by_asset: dict[int, int] = {}
    sql = (
        "SELECT af.asset_id as asset_id, af.phash as phash "
        "FROM asset_faces af JOIN assets a ON a.id = af.asset_id "
        "WHERE a.event_id = ?"
    )
    for row in conn.execute(sql, (int(event_id),)):
        aid = int(row["asset_id"])
        dist = hamming_distance(query_phash, int(row["phash"]))
        if distance_threshold is not None and dist > distance_threshold:
            continue
        prev = best_by_asset.get(aid)
        if prev is None or dist < prev:
            best_by_asset[aid] = dist
    if not best_by_asset:
        return []
    ids = list(best_by_asset.keys())
    placeholders = ",".join(["?"] * len(ids))
    rows = list(conn.execute(f"SELECT id, path FROM assets WHERE id IN ({placeholders})", ids))
    path_by_id = {int(r["id"]): r["path"] for r in rows}
    ranked = [RankedImage(image_path=path_by_id.get(aid, ""), best_distance=dist) for aid, dist in best_by_asset.items() if aid in path_by_id]
    ranked.sort(key=lambda r: r.best_distance)
    return ranked[:max_results]


# ------------------------------- Links & Logs ------------------------------

def create_guest_link(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    created_by_user_id: Optional[int] = None,
    expires_at: Optional[_dt.datetime] = None,
) -> str:
    token = secrets.token_urlsafe(24)
    conn.execute(
        "INSERT INTO guest_links(event_id, token, created_by_user_id, expires_at) VALUES(?, ?, ?, ?)",
        (int(event_id), token, created_by_user_id, expires_at.isoformat() if expires_at else None),
    )
    conn.commit()
    return token


def get_guest_link(conn: sqlite3.Connection, token: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM guest_links WHERE token = ?", (token,))
    return cur.fetchone()


def log_activity(conn: sqlite3.Connection, *, user_id: Optional[int], action: str, detail: Optional[str] = None) -> None:
    conn.execute("INSERT INTO activity_logs(user_id, action, detail) VALUES(?, ?, ?)", (user_id, action, detail))
    conn.commit()


def log_download(conn: sqlite3.Connection, *, user_id: Optional[int], asset_count: int) -> None:
    conn.execute("INSERT INTO downloads(user_id, asset_count) VALUES(?, ?)", (user_id, int(asset_count)))
    conn.commit()


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
