"""Resolve <|ASIN_...|> tokens to human-readable titles."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

ASIN_TOKEN_RE = re.compile(r"<\|ASIN_([^|]+)\|>")


def parse_prediction_tokens(response: str) -> List[str]:
    """Extract ordered ASIN special tokens and <|endoftext|> from model output."""
    if not response or not response.strip():
        return []
    return re.findall(r"<\|ASIN_[^|]+\|>|<\|endoftext\|>", response)


@lru_cache(maxsize=1)
def load_asin_title_map(processed_dir: Path) -> Dict[str, str]:
    """Load parent_asin -> title mapping from processed data."""
    path = processed_dir / "parent_asin_title.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[str, str] = {}
    for row in rows:
        pid = row.get("parent_asin")
        title = row.get("title")
        if pid and title:
            out[pid] = title
    return out


def token_to_catalog_key(token: str) -> Optional[str]:
    if token == "<|endoftext|>":
        return None
    m = ASIN_TOKEN_RE.match(token.strip())
    if not m:
        return None
    return f"<|ASIN_{m.group(1)}|>"


def resolve_tokens(
    tokens: List[str], catalog: Dict[str, str]
) -> List[tuple[str, str]]:
    """Return (token, label) for display."""
    rows: List[tuple[str, str]] = []
    for t in tokens:
        t = t.strip()
        if t == "<|endoftext|>":
            rows.append((t, "(no further purchase — end of sequence)"))
            continue
        key = token_to_catalog_key(t)
        if key and key in catalog:
            rows.append((t, catalog[key]))
        elif key:
            rows.append((t, "Title not in catalog"))
        else:
            rows.append((t, t))
    return rows
