# Billboard-composite-service.
Perspective-warp compositing service for placing ad creatives onto billboard/OOH site photos — part of an n8n mockup automation pipeline.
# Billboard Composite Service

A small web service that places an ad design onto a photo of a billboard or
OOH (out-of-home) advertising site, matching the site's real perspective —
so the design looks like it's actually mounted there, not pasted flat on top.

Built for an AI-powered billboard/OOH mockup generator. This service handles
the actual image compositing step; corner detection, the billboard library,
and the client-facing form all live in a separate n8n workflow that calls
this service over HTTP.

## How it works

Given:
- A base photo of the billboard (by URL)
- The four pixel corners of the billboard's ad-facing surface in that photo
- A design image to place there

...it mathematically warps the design to fit that exact shape (a perspective
transform / homography), then blends it into the original photo — so the
result respects the billboard's real tilt, angle, and perspective instead of
sitting on top as a flat rectangle.

## Endpoints

### `POST /composite`

Generates the composited mockup.

**Form fields:**
| Field | Type | Description |
|---|---|---|
| `base_photo_url` | text | URL of the billboard reference photo |
| `corners` | text (JSON) | `{"TL":[x,y],"TR":[x,y],"BR":[x,y],"BL":[x,y]}` — pixel coordinates of the billboard's four corners in the base photo |
| `design` | file | The ad creative/design image to place on the billboard |

**Response:** the composited image, returned as `image/png`.

### `GET /health`

Simple health check. Returns `{"status": "ok"}` if the service is running.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

Deployed on [Render](https://render.com) as a Python web service:
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Note: on Render's free tier, the service sleeps after periods of inactivity —
the first request after sleeping can take 30–50 seconds to respond.

## Part of a larger project

This service is one piece of an n8n-based automation:
1. Client uploads a new billboard photo → an AI vision step detects its four
   corners → saved to a billboard library (Google Sheet).
2. Client uploads a design and picks a billboard site → this service
   composites the design onto that site's photo → the mockup is delivered
   back to the client.
