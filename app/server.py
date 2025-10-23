from __future__ import annotations

import io
import os
import zipfile
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
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
    create_user,
    get_user_by_email,
    set_user_embedding_phash,
    create_or_get_event,
    upsert_asset,
    insert_asset_face,
    rank_assets_by_phash,
    get_event_by_name,
    rank_assets_by_phash_for_event,
    create_guest_link,
)
from .face import detect_faces_bgr, crop_with_margin, compute_phash
from .auth import hash_password, verify_password, create_access_token, decode_access_token

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
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    return payload


@app.post("/signup")
async def signup(name: str = Form(...), age: int | None = Form(None), gender: str | None = Form(None), contact: str | None = Form(None), email: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    ensure_schema(conn)
    if get_user_by_email(conn, email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = create_user(conn, name=name, age=age, gender=gender, contact=contact, email=email, password_hash=hash_password(password), role="customer")
    token = create_access_token({"sub": str(user_id), "email": email})
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    user = get_user_by_email(conn, email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user["id"]), "email": user["email"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "user_id": user["id"]}


@app.post("/upload")
async def upload(event_name: str = Form(...), files: list[UploadFile] = File(...), user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = get_connection()
    ensure_schema(conn)
    event_id = create_or_get_event(conn, name=event_name, owner_user_id=int(user.get("sub")))
    os.makedirs(GALLERY_DIR, exist_ok=True)
    # Save and index
    for f in files:
        data = await f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # save under event folder
        rel_dir = os.path.join(event_name)
        abs_dir = os.path.join(GALLERY_DIR, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        rel_path = os.path.join(rel_dir, f.filename)
        abs_path = os.path.join(abs_dir, f.filename)
        cv2.imencode('.jpg', img)[1].tofile(abs_path)
        asset_id = upsert_asset(conn, event_id=event_id, rel_path=rel_path)
        # detect faces and index
        faces = detect_faces_bgr(img)
        for box in faces:
            crop = crop_with_margin(img, box, margin_ratio=0.25)
            ph = compute_phash(crop)
            insert_asset_face(conn, asset_id=asset_id, x=box.x, y=box.y, w=box.w, h=box.h, phash=ph)
    return {"status": "ok"}


@app.post("/match_face")
async def match_face(file: UploadFile = File(...), event_name: str | None = Form(None)):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    faces = detect_faces_bgr(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")
    crop = crop_with_margin(img, faces[0], margin_ratio=0.25)
    q = compute_phash(crop)
    conn = get_connection()
    if event_name:
        ev = get_event_by_name(conn, event_name)
        if ev:
            res = rank_assets_by_phash_for_event(conn, q, event_id=int(ev["id"]), max_results=200)
        else:
            res = rank_assets_by_phash(conn, q, max_results=200)
    else:
        res = rank_assets_by_phash(conn, q, max_results=200)
    return {"results": [{"image_path": r.image_path, "distance": r.best_distance} for r in res]}


@app.get("/photos/{user_id}")
async def photos(user_id: int):
    # Placeholder: In a full system, map user embedding to assets via joins
    return JSONResponse({"results": []})


@app.post("/generate_link")
async def generate_link(event_name: str = Form(...), user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = get_connection()
    ev = get_event_by_name(conn, event_name)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    token = create_guest_link(conn, event_id=int(ev["id"]), created_by_user_id=int(user.get("sub")))
    return {"token": token}


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
