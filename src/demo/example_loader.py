"""
Load demo gallery rows from processed data: only examples with a real on-disk local image.

No train/val/test labels are attached — the UI treats them uniformly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _dedupe_key(row: Dict[str, Any]) -> str:
    return f"{row.get('instruction', '')}|{row.get('input', '')}"


def row_has_valid_local_image(row: Dict[str, Any], repo_root: Path) -> bool:
    lip = row.get("local_image_path")
    if not lip:
        return False
    p = Path(str(lip))
    if not p.is_absolute():
        p = repo_root / p
    return p.is_file()


def _to_example(row: Dict[str, Any], model_response: Optional[str] = None) -> Dict[str, Any]:
    # Intentionally no train/val/test split field — the gallery is one list for end users.
    return {
        "title": row.get("title"),
        "main_category": row.get("main_category"),
        "rating": row.get("rating"),
        "review_title": row.get("review_title"),
        "instruction": row.get("instruction"),
        "input": row.get("input"),
        "output": row.get("output"),
        "output_names": row.get("output_names"),
        "model_response": model_response if model_response is not None else row.get("model_response"),
        "image_url": row.get("image_url"),
        "local_image_path": row.get("local_image_path"),
    }


def _load_test_model_responses_by_key(processed_dir: Path) -> Dict[str, str]:
    path = processed_dir / "test_with_responses.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[str, str] = {}
    for row in rows:
        k = _dedupe_key(row)
        mr = row.get("model_response")
        if mr:
            out[k] = mr
    return out


def load_gallery_examples(
    repo_root: Path,
    *,
    max_examples: int = 36,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Prefer processed_data.jsonl (rows with a real local_image_path file).
    If jsonl is missing, fall back to demo/demo_examples.json with the same image filter.
    """
    proc = repo_root / "data" / "processed"
    jsonl_path = proc / "processed_data.jsonl"
    responses = _load_test_model_responses_by_key(proc)
    by_key: Dict[str, Dict[str, Any]] = {}

    if jsonl_path.is_file():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row_has_valid_local_image(row, repo_root):
                    continue
                k = _dedupe_key(row)
                if k in by_key:
                    continue
                mr = responses.get(k)
                by_key[k] = _to_example(row, model_response=mr)
    else:
        demo = repo_root / "demo" / "demo_examples.json"
        if demo.is_file():
            with open(demo, encoding="utf-8") as f:
                rows = json.load(f)
            for row in rows:
                if not row_has_valid_local_image(row, repo_root):
                    continue
                k = _dedupe_key(row)
                if k in by_key:
                    continue
                by_key[k] = _to_example(row)

    pool = list(by_key.values())
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_examples]
