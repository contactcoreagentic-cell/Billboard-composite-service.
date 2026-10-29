"""
Billboard mockup compositing microservice.
Takes a base billboard photo (by URL), a design image (uploaded), and 4 corner
points, and returns the design perspective-warped onto the billboard photo.
Run locally:
    pip install fastapi uvicorn opencv-python-headless numpy python-multipart requests
    uvicorn main:app --host 0.0.0.0 --port 8000
Deploy: push this folder to Railway/Render as a Python web service.
Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import io
import json

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps

app = FastAPI()


# ---------- image loading ----------

def pil_to_cv(img: Image.Image, keep_alpha: bool) -> np.ndarray:
    # Apply the phone's EXIF rotation so the pixels match what a browser shows.
    # (OpenCV's IMREAD_UNCHANGED ignores EXIF, which can flip phone photos.)
    img = ImageOps.exif_transpose(img)
    if keep_alpha:
        arr = np.array(img.convert("RGBA"))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_image_from_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return pil_to_cv(Image.open(io.BytesIO(resp.content)), keep_alpha=False)


def load_image_from_bytes(data: bytes, keep_alpha: bool = False) -> np.ndarray:
    return pil_to_cv(Image.open(io.BytesIO(data)), keep_alpha=keep_alpha)


# ---------- corner handling ----------

def parse_quad(quad) -> np.ndarray:
    """Accepts {"TL":[x,y],"TR":..,"BR":..,"BL":..} or [[x,y],[x,y],[x,y],[x,y]]."""
    if isinstance(quad, dict):
        pts = [quad[k] for k in ("TL", "TR", "BR", "BL")]
    else:
        pts = quad
    arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if arr.shape != (4, 2):
        raise ValueError("each billboard face needs exactly 4 corner points")
    return arr


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Re-label 4 points as TL, TR, BR, BL by their position in the photo,
    ignoring whatever labels they arrived with. Stops upside-down / mirrored
    results when the saved labels are in the wrong order."""
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]            # clockwise on screen: TL, TR, BR, BL
    start = int(np.argmin(pts.sum(axis=1)))  # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)


def parse_faces(faces: str | None, corners: str | None) -> list[np.ndarray]:
    if faces:
        data = json.loads(faces)
        if isinstance(data, dict):  # a single face sent as an object
            data = [data]
    elif corners:
        data = [json.loads(corners)]
    else:
        raise ValueError("send either 'faces' or 'corners'")
    if not data:
        raise ValueError("no billboard faces given")
    return [order_corners(parse_quad(q)) for q in data]


# ---------- compositing ----------

def composite_faces(base: np.ndarray, design: np.ndarray, faces: list[np.ndarray]) -> np.ndarray:
    if design.shape[2] == 4:
        design_bgr = design[:, :, :3]
        design_alpha = design[:, :, 3]
    else:
        design_bgr = design
        design_alpha = np.full(design.shape[:2], 255, dtype=np.uint8)

    h_base, w_base = base.shape[:2]
    h_d, w_d = design_bgr.shape[:2]
    src = np.array([[0, 0], [w_d, 0], [w_d, h_d], [0, h_d]], dtype=np.float32)

    result = base.astype(np.float32)
    for dst in faces:
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(design_bgr, H, (w_base, h_base), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpPerspective(design_alpha, H, (w_base, h_base), flags=cv2.INTER_LINEAR)
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        result = warped.astype(np.float32) * a + result * (1.0 - a)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------- endpoints ----------

@app.post("/composite")
async def composite(
    base_photo_url: str = Form(...),
    design: UploadFile = File(...),
    faces: str | None = Form(None),    # new: JSON list of faces (multi-billboard)
    corners: str | None = Form(None),  # old: single face, still supported
):
    try:
        quads = parse_faces(faces, corners)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"bad corners: {e}")

    base = load_image_from_url(base_photo_url)
    design_img = load_image_from_bytes(await design.read(), keep_alpha=True)

    result = composite_faces(base, design_img, quads)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode result image")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
