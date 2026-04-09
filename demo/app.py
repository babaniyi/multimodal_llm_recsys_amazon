"""
MM-GPT2Rec interactive demo: curated Amazon examples + optional live inference.

Run from the repository root:

    pip install streamlit
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _inject_demo_styles() -> None:
    st.markdown(
        """
<style>
  .demo-no-photo {
    min-height: 140px; display: flex; align-items: center; justify-content: center;
    border-radius: 10px; margin-bottom: 0.5rem; font-size: 0.88rem;
    border: 1px dashed rgba(128,128,128,0.35); color: rgba(128,128,128,0.85);
  }
  .demo-stop-card {
    min-height: 100px; display: flex; align-items: center; justify-content: center;
    border-radius: 10px; margin-bottom: 0.5rem; font-weight: 600; font-size: 0.95rem;
    border: 1px solid rgba(99, 102, 241, 0.35); color: inherit;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _gallery_file_fingerprint() -> Tuple[float, float]:
    proc = REPO_ROOT / "data" / "processed"
    jl = proc / "processed_data.jsonl"
    tr = proc / "test_with_responses.json"
    return (
        jl.stat().st_mtime if jl.is_file() else 0.0,
        tr.stat().st_mtime if tr.is_file() else 0.0,
    )


@st.cache_data(show_spinner="Loading examples…")
def _cached_gallery_examples(
    repo_root_str: str,
    jsonl_mtime: float,
    test_mtime: float,
    max_examples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    from src.demo.example_loader import load_gallery_examples

    return load_gallery_examples(
        Path(repo_root_str),
        max_examples=max_examples,
        seed=seed,
    )


@st.cache_data(show_spinner="Loading catalog…")
def _title_catalog() -> Dict[str, str]:
    path = REPO_ROOT / "data" / "processed" / "parent_asin_title.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {
        row["parent_asin"]: row["title"]
        for row in rows
        if row.get("parent_asin") and row.get("title")
    }


def _checkpoint_mtime(path_str: str) -> float:
    p = Path(path_str)
    if p.is_file():
        return p.stat().st_mtime
    return 0.0


@st.cache_resource(show_spinner="Loading recommendation model…")
def _load_inference_bundle(checkpoint_path_str: str, checkpoint_mtime: float):
    from src.demo.inference import build_model_for_demo, load_fine_tuned_weights
    from src.utils.utils import load_config

    config = load_config(str(REPO_ROOT / "configs" / "train_config.json"))
    model, tokenizer, special_chars, end_id, base_cfg, _cfg = build_model_for_demo(
        REPO_ROOT, config=config
    )

    device = next(model.parameters()).device
    ckpt = Path(checkpoint_path_str)
    if ckpt.is_file():
        load_fine_tuned_weights(model, ckpt, str(device))
    return model, tokenizer, special_chars, end_id, base_cfg, config, ckpt.is_file()


def _anchor_image_path(ex: Dict[str, Any]) -> Optional[Path]:
    lip = ex.get("local_image_path")
    if not lip:
        return None
    p = Path(str(lip))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p if p.is_file() else None


def main():
    st.set_page_config(
        page_title="Product recommendations",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_demo_styles()

    st.title("Product recommendations")
    st.markdown(
        "Pick a shopping story, see **what they bought next**, and compare with **what the model suggests**. "
        "Use the sidebar to load the model when you want new suggestions."
    )

    catalog = _title_catalog()
    jl_mt, test_mt = _gallery_file_fingerprint()
    examples = _cached_gallery_examples(
        str(REPO_ROOT), jl_mt, test_mt, max_examples=36, seed=42
    )

    with st.sidebar:
        st.header("Recommendation model")
        st.caption("Load your trained model file to generate new suggestions for any example.")
        default_ckpt = REPO_ROOT / "output" / "model" / "finetuned-gpt2-medium-sft.pth"
        ckpt_input = st.text_input(
            "Model file",
            value=str(default_ckpt),
            help="Trained weights saved after running the training script.",
        )

        if (
            st.session_state.get("bundle") is None
            and Path(ckpt_input).is_file()
            and st.session_state.get("_demo_auto_fail_path") != ckpt_input
        ):
            try:
                st.session_state["bundle"] = _load_inference_bundle(
                    ckpt_input, _checkpoint_mtime(ckpt_input)
                )
                st.session_state["_demo_auto_loaded"] = True
                st.session_state.pop("_demo_auto_fail_path", None)
            except Exception as e:
                st.session_state["_demo_auto_fail_path"] = ckpt_input
                st.session_state["_demo_auto_load_error"] = str(e)

        if st.session_state.pop("_demo_auto_loaded", False):
            st.success("Model loaded.")
        err = st.session_state.pop("_demo_auto_load_error", None)
        if err:
            st.warning(f"Could not load automatically. Try the button below.\n{err}")

        if st.button("Load / reload model", use_container_width=True):
            try:
                bundle = _load_inference_bundle(ckpt_input, _checkpoint_mtime(ckpt_input))
                st.session_state["bundle"] = bundle
                st.session_state.pop("_demo_auto_fail_path", None)
                _, _, _, _, _, _, ok = bundle
                if ok:
                    st.success("Model ready.")
                else:
                    st.warning(
                        "Model file not found — using a generic base. Add your trained file for real suggestions."
                    )
            except Exception as e:
                st.error(f"Failed: {e}")
                st.session_state.pop("bundle", None)

        st.divider()
        if st.button("Refresh example list", use_container_width=True):
            _cached_gallery_examples.clear()
            st.rerun()

        st.caption(f"{len(examples)} examples · {len(catalog):,} products in catalog")

    tab_gallery, tab_custom = st.tabs(["Examples", "Try your own"])

    with tab_gallery:
        if not examples:
            st.error(
                "No examples with product photos were found. "
                "Check that images exist under `data/processed/image_cache/` and paths in `processed_data.jsonl` are valid."
            )
        else:
            from src.demo.item_cards import build_item_cards, render_item_cards_streamlit

            def _example_label(i: int) -> str:
                title = examples[i].get("title") or "Product"
                tail = "…" if len(title) > 76 else ""
                return f"{i + 1}. {title[:76]}{tail}"

            labels = [_example_label(i) for i in range(len(examples))]
            st.caption("**1 · Pick a story**")
            idx = st.selectbox(
                "Choose an example",
                range(len(examples)),
                format_func=lambda i: labels[i],
            )
            ex = examples[idx]

            st.markdown("---")
            st.caption("**2 · Product they’re considering**")

            img_col, text_col = st.columns([1, 1.85], gap="large")
            with img_col:
                local_p = _anchor_image_path(ex)
                if local_p is not None:
                    st.image(str(local_p), use_container_width=True)
                elif ex.get("image_url"):
                    st.image(ex["image_url"], use_container_width=True)
                else:
                    st.markdown('<div class="demo-no-photo">No photo</div>', unsafe_allow_html=True)
                st.metric("Rating", f"{ex.get('rating', '—')} ★" if ex.get("rating") is not None else "—")
            with text_col:
                cat = ex.get("main_category") or ""
                if cat:
                    st.caption(cat)
                st.markdown(f"## {ex.get('title') or 'Product'}")
                if ex.get("review_title"):
                    st.markdown(f"> *{ex['review_title']}*")

            st.markdown("---")
            st.caption("**3 · What happened next**")

            gt = ex.get("output") or ""
            cached = (ex.get("model_response") or "").strip()
            gt_cards = build_item_cards(gt, ex.get("output_names"), catalog, REPO_ROOT)

            left, right = st.columns(2, gap="large")
            with left:
                render_item_cards_streamlit(gt_cards, "What they bought next")
            with right:
                live_s = (st.session_state.get(f"live_{idx}") or "").strip()
                if live_s:
                    render_item_cards_streamlit(
                        build_item_cards(live_s, None, catalog, REPO_ROOT),
                        "What we’d suggest · just generated",
                    )
                elif cached:
                    render_item_cards_streamlit(
                        build_item_cards(cached, None, catalog, REPO_ROOT),
                        "What we’d suggest",
                    )
                else:
                    st.markdown("#### What we’d suggest")
                    st.info(
                        "Load the model in the sidebar, then use the button below to generate suggestions for this example."
                    )

            bundle = st.session_state.get("bundle")
            if bundle and st.button("Generate suggestions for this example", key=f"regen_{idx}"):
                model, tokenizer, special_chars, end_id, base_cfg, config, _ok = bundle
                from src.demo.inference import generate_recommendations

                device = str(next(model.parameters()).device)
                entry = {
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                }
                with st.spinner("Working…"):
                    try:
                        live = generate_recommendations(
                            model,
                            tokenizer,
                            special_chars,
                            end_id,
                            base_cfg,
                            entry,
                            config["training_config"],
                            device,
                        )
                        st.session_state[f"live_{idx}"] = live
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    with tab_custom:
        st.markdown(
            "Describe a shopper and product the same way as in the examples. The model must already be loaded in the sidebar."
        )
        default_instr = (
            examples[0].get("instruction", "")
            if examples
            else "Given a user purchased an item with the following details, predict the next 5 items the user would purchase. "
        )
        instr = st.text_area("Instructions for the model", value=default_instr, height=88)
        user_input = st.text_area(
            "Shopper and product details",
            value=examples[0]["input"] if examples else "",
            height=140,
        )
        bundle = st.session_state.get("bundle")
        if st.button("Get suggestions", disabled=bundle is None):
            from src.demo.item_cards import build_item_cards, render_item_cards_streamlit
            from src.demo.inference import generate_recommendations

            model, tokenizer, special_chars, end_id, base_cfg, config, ok = bundle
            device = str(next(model.parameters()).device)
            entry = {"instruction": instr, "input": user_input}
            with st.spinner("Working…"):
                try:
                    out = generate_recommendations(
                        model,
                        tokenizer,
                        special_chars,
                        end_id,
                        base_cfg,
                        entry,
                        config["training_config"],
                        device,
                    )
                    render_item_cards_streamlit(
                        build_item_cards(out, None, catalog, REPO_ROOT),
                        "Suggestions",
                    )
                except Exception as e:
                    st.error(str(e))
        elif bundle is None:
            st.info("Open the sidebar and load the recommendation model first.")

    st.divider()
    st.caption("MM-GPT2Rec · research demo")


if __name__ == "__main__":
    main()
