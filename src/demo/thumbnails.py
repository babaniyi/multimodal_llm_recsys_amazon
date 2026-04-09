"""Fixed square thumbnails so hero, ground truth, and model tiles share identical dimensions."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

# Single size for every product photo in the demo (width == height in pixels).
THUMB_SIZE = 220


def _square_cover_pil(img: Image.Image, side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image dimensions")
    scale = max(side / w, side / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - side) // 2
    top = (nh - side) // 2
    return img.crop((left, top, left + side, top + side))


def _to_png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def thumbnail_bytes_from_path(path: Path) -> Optional[bytes]:
    try:
        with Image.open(path) as im:
            sq = _square_cover_pil(im, THUMB_SIZE)
        return _to_png_bytes(sq)
    except Exception:
        return None


def thumbnail_bytes_from_url(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DemoBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with Image.open(io.BytesIO(data)) as im:
            sq = _square_cover_pil(im, THUMB_SIZE)
        return _to_png_bytes(sq)
    except Exception:
        return None


def placeholder_thumbnail_bytes(hint: Optional[str] = None) -> bytes:
    im = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (36, 38, 46))
    if hint:
        draw = ImageDraw.Draw(im)
        t = hint if len(hint) <= 24 else hint[:21] + "…"
        draw.text((12, THUMB_SIZE // 2 - 8), t, fill=(120, 125, 140))
    return _to_png_bytes(im)


def show_uniform_product_image(
    st: object,
    *,
    local_path: Optional[Path] = None,
    image_url: Optional[str] = None,
) -> None:
    """Display a THUMB_SIZE×THUMB_SIZE image in Streamlit (local file, then URL, else placeholder)."""
    b: Optional[bytes] = None
    if local_path is not None and local_path.is_file():
        b = thumbnail_bytes_from_path(local_path)
    if b is None and image_url:
        b = thumbnail_bytes_from_url(str(image_url))
    if b is None:
        b = placeholder_thumbnail_bytes("No photo")
    st.image(b, width=THUMB_SIZE)


def show_uniform_card_image(st: object, card: dict) -> None:
    """Render card image or same-size placeholder (including end-of-list tile)."""
    tok = (card.get("token") or "").strip()
    if tok == "<|endoftext|>":
        st.image(placeholder_thumbnail_bytes("End of list"), width=THUMB_SIZE)
        return
    path = card.get("image_path")
    if card.get("has_image") and path:
        b = thumbnail_bytes_from_path(Path(path))
        if b:
            st.image(b, width=THUMB_SIZE)
            return
    st.image(placeholder_thumbnail_bytes("No photo"), width=THUMB_SIZE)
