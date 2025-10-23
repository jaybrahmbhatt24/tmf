from __future__ import annotations

import io
import os
import zipfile
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import numpy as np
import cv2

from .db import (
    get_connection,
    ensure_schema,
    rank_by_phash,
    GALLERY_DIR,
    relpath_from_gallery,
    abspath_in_gallery,
)
from .face import detect_faces_bgr, crop_with_margin, compute_phash

app = FastAPI(title="THE MOVING FRAMES")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
if os.path.isdir(GALLERY_DIR):
    # Mount gallery for direct image serving in dev
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")


@app.on_event("startup")
def on_startup() -> None:
    conn = get_connection()
    ensure_schema(conn)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "gallery": GALLERY_DIR})


@app.post("/search")
async def search(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file_bytes = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image upload")
    finally:
        await file.close()

    faces = detect_faces_bgr(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    # Use the largest face
    face_crop = crop_with_margin(img, faces[0], margin_ratio=0.25)
    query_phash = compute_phash(face_crop)

    conn = get_connection()
    results = rank_by_phash(conn, query_phash, max_results=100)

    # Render results
    return {"results": [
        {"image_path": relpath_from_gallery(r.image_path), "distance": r.best_distance}
        for r in results
    ]}


@app.post("/download")
async def download(selected: List[str] = Form(...)):
    # selected contains relative paths
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in selected:
            abs_path = abspath_in_gallery(rel_path)
            if os.path.isfile(abs_path):
                zf.write(abs_path, arcname=rel_path)
    buf.seek(0)
    headers = {"Content-Disposition": "attachment; filename=tmf_photos.zip"}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
