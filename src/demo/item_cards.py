"""Map recommendation strings to titles and optional cache images (no raw IDs in the UI)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.demo.catalog import parse_prediction_tokens


def asin_code_from_special_token(token: str) -> Optional[str]:
    m = re.match(r"<\|ASIN_([^|]+)\|>", token.strip())
    return m.group(1) if m else None


def parse_output_name_list(output_names: Optional[str]) -> List[str]:
    if not output_names or not str(output_names).strip():
        return []
    parts = re.split(r"\s*\|\s*", str(output_names).strip())
    return [p.strip() for p in parts if p.strip()]


def build_item_cards(
    output: str,
    output_names: Optional[str],
    catalog: Dict[str, str],
    repo_root: Path,
) -> List[Dict[str, Any]]:
    tokens = parse_prediction_tokens(output)
    names = parse_output_name_list(output_names)
    cache_dir = repo_root / "data" / "processed" / "image_cache"
    cards: List[Dict[str, Any]] = []
    for i, tok in enumerate(tokens):
        if tok.strip() == "<|endoftext|>":
            cards.append(
                {
                    "token": tok,
                    "title": "End of list",
                    "subtitle": "No further item in this suggestion",
                    "image_path": None,
                    "has_image": False,
                    "asin_code": None,
                }
            )
            continue
        raw_title = names[i] if i < len(names) else catalog.get(tok)
        title = raw_title if raw_title else "Product name unavailable"
        code = asin_code_from_special_token(tok)
        img_path: Optional[Path] = None
        has_img = False
        if code:
            candidate = cache_dir / f"{code}.jpg"
            if candidate.is_file():
                img_path = candidate
                has_img = True
        cards.append(
            {
                "token": tok,
                "title": title,
                "subtitle": None,
                "image_path": img_path,
                "has_image": has_img,
                "asin_code": code,
            }
        )
    return cards


def render_item_cards_streamlit(cards: List[Dict[str, Any]], heading: str) -> None:
    import streamlit as st

    st.markdown(f'<p class="demo-rail-title">{heading}</p>', unsafe_allow_html=True)
    if not cards:
        st.caption("Nothing to show here.")
        return
    chunk_size = 5
    for start in range(0, len(cards), chunk_size):
        chunk = cards[start : start + chunk_size]
        cols = st.columns(len(chunk))
        for col, card in zip(cols, chunk):
            with col:
                with st.container(border=True):
                    if card.get("has_image") and card.get("image_path"):
                        st.image(str(card["image_path"]), use_container_width=True)
                    elif card["token"].strip() != "<|endoftext|>":
                        st.markdown(
                            '<div class="demo-img-placeholder">Photo not available</div>',
                            unsafe_allow_html=True,
                        )
                    title = card.get("title") or ""
                    short = title if len(title) <= 110 else title[:107] + "…"
                    st.markdown(
                        f'<p class="demo-card-title">{html.escape(short)}</p>',
                        unsafe_allow_html=True,
                    )
                    sub = card.get("subtitle")
                    if sub:
                        st.caption(sub)
