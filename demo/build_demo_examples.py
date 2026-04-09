#!/usr/bin/env python3
"""Write demo/demo_examples.json from test_with_responses.json (run from repo root)."""

import json
import random
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "data" / "processed" / "test_with_responses.json"
    if not src.is_file():
        print(f"Missing {src}", file=sys.stderr)
        sys.exit(1)
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(42)
    with_img = [e for e in data if e.get("local_image_path")]
    pool = with_img if len(with_img) >= 8 else data
    picks = rng.sample(pool, min(12, len(pool)))

    def slim(e):
        return {
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

    out_path = root / "demo" / "demo_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([slim(e) for e in picks], f, indent=2)
    print(f"Wrote {len(picks)} examples to {out_path}")


if __name__ == "__main__":
    main()
