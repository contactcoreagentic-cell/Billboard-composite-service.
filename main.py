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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()


def load_image_from_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_image_from_bytes(data: bytes, keep_alpha: bool = False) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    flag = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    return cv2.imdecode(arr, flag)


@app.post("/composite")
async def composite(
    base_photo_url: str = Form(...),
    corners: str = Form(...),  # JSON string: {"TL":[x,y],"TR":[x,y],"BR":[x,y],"BL":[x,y]}
    design: UploadFile = File(...),
):
    base = load_image_from_url(base_photo_url)
        design_bytes = await design.read()
    design_img = load_image_from_bytes(design_bytes, keep_alpha=True)

    if design_img.shape[2] == 4:
        design_bgr = design_img[:, :, :3]
        design_alpha = design_img[:, :, 3]
    else:
        design_bgr = design_img
        design_alpha = np.full(design_img.shape[:2], 255, dtype=np.uint8)

    pts = json.loads(corners)
    h_base, w_base = base.shape[:2]
    h_design, w_design = design_bgr.shape[:2]
    dst_pts = np.array(
        [pts["TL"], pts["TR"], pts["BR"], pts["BL"]], dtype=np.float32
    )
    src_pts = np.array(
        [[0, 0], [w_design, 0], [w_design, h_design], [0, h_design]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(design_bgr, H, (w_base, h_base))
    warped_alpha = cv2.warpPerspective(design_alpha, H, (w_base, h_base))
    mask = warped_alpha
    mask_inv = cv2.bitwise_not(mask)
    base_bg = cv2.bitwise_and(base, base, mask=mask_inv)
    warped_fg = cv2.bitwise_and(warped, warped, mask=mask)
    result = cv2.add(base_bg, warped_fg)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        return {"error": "failed to encode result image"}

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
