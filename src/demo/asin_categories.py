"""Map <|ASIN_…|> tokens to Amazon main_category strings from processed rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_parent_asin_category_map(repo_root: Path) -> Dict[str, str]:
    """
    Build parent_asin -> main_category from processed_data.jsonl and split JSON files.
    Later rows overwrite earlier ones if the same ASIN appears again.
    """
    proc = repo_root / "data" / "processed"
    out: Dict[str, str] = {}

    jsonl = proc / "processed_data.jsonl"
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                _ingest_row(out, row)

    for name in ("train.json", "val.json", "test.json"):
        p = proc / name
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            rows: List[Dict[str, Any]] = json.load(f)
        for row in rows:
            _ingest_row(out, row)

    return out


def _ingest_row(out: Dict[str, str], row: Dict[str, Any]) -> None:
    pa = row.get("parent_asin")
    mc = row.get("main_category")
    if not isinstance(pa, str) or not pa.strip():
        return
    if not isinstance(mc, str) or not mc.strip():
        return
    out[pa.strip()] = mc.strip()
