"""Map recommendation strings to human-friendly titles and local product images."""

from __future__ import annotations

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
                    "title": "No further purchase in this period",
                    "image_path": None,
                    "has_image": False,
                    "asin_code": None,
                    "is_stop": True,
                }
            )
            continue
        title = names[i] if i < len(names) else catalog.get(tok, "Product")
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
                "image_path": img_path,
                "has_image": has_img,
                "asin_code": code,
                "is_stop": False,
            }
        )
    return cards


def render_item_cards_streamlit(cards: List[Dict[str, Any]], heading: str) -> None:
    import streamlit as st

    st.markdown(f"#### {heading}")
    if not cards:
        st.caption("Nothing to show for this example.")
        return
    chunk_size = 5
    for start in range(0, len(cards), chunk_size):
        chunk = cards[start : start + chunk_size]
        cols = st.columns(len(chunk), gap="small")
        for col, card in zip(cols, chunk):
            with col:
                if card.get("is_stop"):
                    st.markdown(
                        '<div class="demo-stop-card">Done shopping</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(card.get("title", ""))
                elif card.get("has_image") and card.get("image_path"):
                    st.image(str(card["image_path"]), use_container_width=True)
                else:
                    st.markdown(
                        '<div class="demo-no-photo">No photo</div>',
                        unsafe_allow_html=True,
                    )
                title = (card.get("title") or "").strip()
                if title and not card.get("is_stop"):
                    short = title if len(title) <= 110 else title[:107] + "…"
                    st.markdown(f"**{short}**")
