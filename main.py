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

FIT_ALIASES = {
    "stretch": "stretch", "": "stretch",
    "cover": "cover", "crop": "cover", "crop to fill": "cover", "fill": "cover",
    "contain": "contain", "fit": "contain", "fit inside": "contain",
}


def normalise_fit(fit: str | None) -> str:
    key = (fit or "").strip().lower()
    if key not in FIT_ALIASES:
        raise ValueError(f"unknown fit '{fit}' (use stretch, cover or contain)")
    return FIT_ALIASES[key]


def face_aspect(quad: np.ndarray) -> float:
    """Approximate width/height of the real board from its 4 corners (TL, TR, BR, BL)."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(width / max(height, 1e-6))


def parse_colour(hex_colour: str | None, design_bgr: np.ndarray) -> np.ndarray:
    """'#RRGGBB' -> BGR. Without one, use the design's typical edge colour."""
    if hex_colour:
        h = hex_colour.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError("background must look like #RRGGBB")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return np.array([b, g, r], dtype=np.uint8)
    edges = np.concatenate([design_bgr[0], design_bgr[-1], design_bgr[:, 0], design_bgr[:, -1]])
    return np.median(edges, axis=0).astype(np.uint8)


def fit_design(bgr: np.ndarray, alpha: np.ndarray, target_aspect: float,
               fit: str, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reshape the design to the board's proportions so it isn't distorted."""
    if fit == "stretch":
        return bgr, alpha
    h, w = bgr.shape[:2]
    aspect = w / h
    if fit == "cover":  # trim the edges, keep the centre
        if aspect > target_aspect:
            new_w = max(1, round(h * target_aspect))
            x0 = (w - new_w) // 2
            return bgr[:, x0:x0 + new_w], alpha[:, x0:x0 + new_w]
        new_h = max(1, round(w / target_aspect))
        y0 = (h - new_h) // 2
        return bgr[y0:y0 + new_h], alpha[y0:y0 + new_h]
    # contain: pad with a plain background, design centred and whole
    if aspect > target_aspect:
        canvas_w, canvas_h = w, max(h, round(w / target_aspect))
    else:
        canvas_w, canvas_h = max(w, round(h * target_aspect)), h
    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = bg
    canvas_alpha = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    x0, y0 = (canvas_w - w) // 2, (canvas_h - h) // 2
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    region = canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)
    canvas[y0:y0 + h, x0:x0 + w] = (bgr * a + region * (1 - a)).astype(np.uint8)
    return canvas, canvas_alpha


def composite_faces(base: np.ndarray, design: np.ndarray, faces: list[np.ndarray],
                    fit: str = "stretch", background: str | None = None) -> np.ndarray:
    if design.shape[2] == 4:
        design_bgr = design[:, :, :3]
        design_alpha = design[:, :, 3]
    else:
        design_bgr = design
        design_alpha = np.full(design.shape[:2], 255, dtype=np.uint8)

    fit = normalise_fit(fit)
    bg = parse_colour(background, design_bgr)
    h_base, w_base = base.shape[:2]

    result = base.astype(np.float32)
    for dst in faces:
        face_bgr, face_alpha = fit_design(design_bgr, design_alpha, face_aspect(dst), fit, bg)
        h_d, w_d = face_bgr.shape[:2]
        src = np.array([[0, 0], [w_d, 0], [w_d, h_d], [0, h_d]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(face_bgr, H, (w_base, h_base), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpPerspective(face_alpha, H, (w_base, h_base), flags=cv2.INTER_LINEAR)
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        result = warped.astype(np.float32) * a + result * (1.0 - a)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------- endpoints ----------

@app.post("/composite")
async def composite(
    base_photo_url: str = Form(...),
    design: UploadFile = File(...),
    faces: str | None = Form(None),       # new: JSON list of faces (multi-billboard)
    corners: str | None = Form(None),     # old: single face, still supported
    fit: str | None = Form(None),         # stretch (default) | cover | contain
    background: str | None = Form(None),  # #RRGGBB for "contain"; default = design's edge colour
):
    try:
        quads = parse_faces(faces, corners)
        fit_mode = normalise_fit(fit)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"bad input: {e}")

    base = load_image_from_url(base_photo_url)
    design_img = load_image_from_bytes(await design.read(), keep_alpha=True)

    try:
        result = composite_faces(base, design_img, quads, fit_mode, background)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"bad input: {e}")

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode result image")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
