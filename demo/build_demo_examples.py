#!/usr/bin/env python3
"""
Optional: export demo/demo_examples.json for offline use.

The Streamlit app normally builds the gallery from processed_data.jsonl at runtime
(see src.demo.example_loader) and does not require this script.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.catalog import parse_prediction_tokens


def _pick(
    pool: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    prefer_image: bool = True,
) -> List[Dict[str, Any]]:
    if n <= 0 or not pool:
        return []
    if prefer_image:
        with_img = [e for e in pool if e.get("image_url") or e.get("local_image_path")]
        use = with_img if len(with_img) >= min(n, 8) else pool
    else:
        use = pool
    k = min(n, len(use))
    return rng.sample(use, k)


def _slim(e: Dict[str, Any], split: str) -> Dict[str, Any]:
    return {
        "split": split,
        "title": e.get("title"),
        "main_category": e.get("main_category"),
        "rating": e.get("rating"),
        "review_title": e.get("review_title"),
        "instruction": e.get("instruction"),
        "input": e.get("input"),
        "output": e.get("output"),
        "output_names": e.get("output_names"),
        "model_response": e.get("model_response"),
        "image_url": e.get("image_url"),
        "local_image_path": e.get("local_image_path"),
    }


def _dedupe_key(row: Dict[str, Any]) -> str:
    return f"{row.get('instruction', '')}|{row.get('input', '')}"


def _output_image_score(row: Dict[str, Any], repo_root: Path) -> int:
    """Count next-item ASINs that have a JPEG in image_cache/."""
    import re

    out = row.get("output") or ""
    cache = repo_root / "data" / "processed" / "image_cache"
    n = 0
    for tok in parse_prediction_tokens(out):
        if tok == "<|endoftext|>":
            continue
        m = re.match(r"<\|ASIN_([^|]+)\|>", tok.strip())
        if m and (cache / f"{m.group(1)}.jpg").is_file():
            n += 1
    return n


def _pick_train_rich(
    pool: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    repo_root: Path,
) -> List[Dict[str, Any]]:
    if not pool or n <= 0:
        return []
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, e in enumerate(pool):
        anchor = 2 if e.get("local_image_path") else 0
        sc = _output_image_score(e, repo_root) * 3 + anchor
        scored.append((sc, i, e))
    scored.sort(key=lambda x: (-x[0], x[1]))
    head_n = min(len(scored), max(n * 8, n))
    head = [t[2] for t in scored[:head_n]]
    rng.shuffle(head)
    return head[:n]


def _load_train_from_jsonl(repo_root: Path, train_keys: Set[str]) -> List[Dict[str, Any]]:
    path = repo_root / "data" / "processed" / "processed_data.jsonl"
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if _dedupe_key(e) in train_keys:
                rows.append(e)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo/demo_examples.json for the Streamlit app.")
    parser.add_argument("--train", type=int, default=14, help="Train samples (from processed_data.jsonl ∩ train)")
    parser.add_argument("--val", type=int, default=6, help="Samples from val.json")
    parser.add_argument("--test", type=int, default=12, help="Samples from test (with cached responses if available)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = ROOT
    proc = root / "data" / "processed"

    train_p = proc / "train.json"
    val_p = proc / "val.json"
    test_resp_p = proc / "test_with_responses.json"
    test_plain_p = proc / "test.json"

    missing = [p for p in (train_p, val_p) if not p.is_file()]
    if missing:
        print(f"Missing required files: {missing}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    with open(train_p, encoding="utf-8") as f:
        train_data: List[Dict[str, Any]] = json.load(f)
    with open(val_p, encoding="utf-8") as f:
        val_data: List[Dict[str, Any]] = json.load(f)

    train_keys = {_dedupe_key(e) for e in train_data}
    train_pool = _load_train_from_jsonl(root, train_keys)
    if len(train_pool) < args.train:
        print(
            f"Warning: only {len(train_pool)} train rows matched processed_data.jsonl; "
            f"filling with train.json.",
            file=sys.stderr,
        )
        train_pool = list(train_data)

    if test_resp_p.is_file():
        with open(test_resp_p, encoding="utf-8") as f:
            test_data = json.load(f)
        test_source = str(test_resp_p.name)
    elif test_plain_p.is_file():
        with open(test_plain_p, encoding="utf-8") as f:
            test_data = json.load(f)
        test_source = str(test_plain_p.name)
    else:
        print(f"Missing {test_resp_p} or {test_plain_p}", file=sys.stderr)
        sys.exit(1)

    picked: List[Tuple[Dict[str, Any], str]] = []
    seen: Set[str] = set()

    def add_batch(rows: List[Dict[str, Any]], split: str) -> None:
        for e in rows:
            k = _dedupe_key(e)
            if k in seen:
                continue
            seen.add(k)
            picked.append((e, split))

    add_batch(_pick_train_rich(train_pool, args.train, rng, root), "train")
    add_batch(_pick(val_data, args.val, rng), "val")
    add_batch(_pick(test_data, args.test, rng), "test")

    out_rows = [_slim(e, split) for e, split in picked]

    out_path = root / "demo" / "demo_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)

    n_cached = sum(1 for r in out_rows if r.get("model_response"))
    print(
        f"Wrote {len(out_rows)} examples to {out_path} "
        f"(train={args.train} from jsonl∩train, val={args.val}, test={args.test}; "
        f"test file={test_source}; {n_cached} with cached model_response)"
    )


if __name__ == "__main__":
    main()
