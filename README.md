# THE MOVING FRAMES

Tagline: "Find Your Moments, Just by Your Face."

Face recognition–style photo retrieval for event photographers. Guests upload a selfie and instantly find photos of themselves from large galleries.

## Quickstart

1) Install system deps (Linux):

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

2) Install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Put event photos under `data/gallery/` (subfolders allowed).

4) Index faces:

```bash
python -m app.cli.index_gallery --gallery data/gallery --event default
```

5) Run the server:

```bash
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and upload a selfie to search. Select results to download a ZIP.

### Auth and Upload

Register a user and upload photos to an event via API (or future dashboard):

```bash
curl -X POST http://localhost:8000/signup -F name=Alice -F email=alice@example.com -F password=pw

TOKEN=$(curl -s -X POST http://localhost:8000/login -F email=alice@example.com -F password=pw | jq -r .access_token)

curl -H "Authorization: Bearer $TOKEN" -F event_name=wedding1 \
     -F files=@/path/to/photo1.jpg -F files=@/path/to/photo2.jpg \
     http://localhost:8000/upload
```

## How it works

- Haar cascade detects faces and crops with margin
- Perceptual hash (pHash via DCT) produces a 64‑bit embedding
- SQLite stores face records per image; retrieval ranks by Hamming distance

## Notes

- For production, serve the gallery images at `/gallery/<relpath>` via your web server mapping to `data/gallery/`.
- Accuracy can be improved by swapping Haar+pHash for a learned face embedding model (e.g., FaceNet, ArcFace).
