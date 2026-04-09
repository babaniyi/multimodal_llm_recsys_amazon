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


def render_one_card_streamlit(card: Dict[str, Any]) -> None:
    """Single product tile (image + title) inside a bordered container."""
    import streamlit as st

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


def render_horizontal_comparison_streamlit(
    gt_cards: List[Dict[str, Any]],
    pred_cards: List[Dict[str, Any]],
    *,
    left_heading: str,
    right_heading: str,
) -> None:
    """
    Each row: [what they bought next | model suggestion] — horizontal pairing by position.
    """
    import streamlit as st

    h0, h1 = st.columns(2, gap="large")
    with h0:
        st.markdown(
            f'<p class="demo-rail-title">{html.escape(left_heading)}</p>',
            unsafe_allow_html=True,
        )
    with h1:
        st.markdown(
            f'<p class="demo-rail-title">{html.escape(right_heading)}</p>',
            unsafe_allow_html=True,
        )

    n = max(len(gt_cards), len(pred_cards))
    if n == 0:
        st.caption("Nothing to compare.")
        return

    for i in range(n):
        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            if i < len(gt_cards):
                render_one_card_streamlit(gt_cards[i])
            else:
                with st.container(border=True):
                    st.caption("—")
        with c_right:
            if i < len(pred_cards):
                render_one_card_streamlit(pred_cards[i])
            else:
                with st.container(border=True):
                    st.caption("—")


def render_item_cards_streamlit(cards: List[Dict[str, Any]], heading: str) -> None:
    import streamlit as st

    st.markdown(
        f'<p class="demo-rail-title">{html.escape(heading)}</p>',
        unsafe_allow_html=True,
    )
    if not cards:
        st.caption("Nothing to show here.")
        return
    chunk_size = 5
    for start in range(0, len(cards), chunk_size):
        chunk = cards[start : start + chunk_size]
        cols = st.columns(len(chunk))
        for col, card in zip(cols, chunk):
            with col:
                render_one_card_streamlit(card)
