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
  .demo-lead { font-size: 1.05rem; line-height: 1.55; color: rgba(250,250,250,0.88); margin-bottom: 1.25rem; max-width: 52rem; }
  .demo-rail-title { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 0.65rem 0;
    padding-bottom: 0.4rem; border-bottom: 1px solid rgba(250,250,250,0.12); color: #fafafa; }
  .demo-card-title { font-size: 0.82rem; font-weight: 500; line-height: 1.35; margin: 0.35rem 0 0 0; color: rgba(250,250,250,0.95); }
  .demo-hero-title { font-size: 1.35rem; font-weight: 650; letter-spacing: -0.03em; line-height: 1.25; margin: 0 0 0.5rem 0; color: #fff; }
  .demo-compare-wrap { margin-top: 0.5rem; }
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

    st.markdown('<p class="demo-hero-title">Product recommendations</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="demo-lead">Each row compares <strong>what the customer bought next</strong> (left) with '
        "<strong>what the model suggested</strong> (right) for the same spot in the list. "
        "Examples below all include saved model suggestions. Load the model in the sidebar to refresh suggestions.</p>",
        unsafe_allow_html=True,
    )

    catalog = _title_catalog()
    jl_mt, test_mt = _gallery_file_fingerprint()
    examples = _cached_gallery_examples(
        str(REPO_ROOT), jl_mt, test_mt, max_examples=36, seed=42
    )

    with st.sidebar:
        st.header("Model")
        st.caption("Load your trained file to run **Get new suggestions** on any example.")
        default_ckpt = REPO_ROOT / "output" / "model" / "finetuned-gpt2-medium-sft.pth"
        st.caption("Trained weights file (.pth)")
        ckpt_input = st.text_input(
            "Model file",
            value=str(default_ckpt),
            label_visibility="collapsed",
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
            st.warning(f"Could not load automatically. Try the button below. ({err})")

        if st.button("Load / reload model", use_container_width=True):
            try:
                bundle = _load_inference_bundle(ckpt_input, _checkpoint_mtime(ckpt_input))
                st.session_state["bundle"] = bundle
                st.session_state.pop("_demo_auto_fail_path", None)
                _, _, _, _, _, _, ok = bundle
                if ok:
                    st.success("Model ready.")
                else:
                    st.warning("File not found — using base model only until you add the trained file.")
            except Exception as e:
                st.error(f"Error: {e}")
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
                "No examples are ready to show. Each example needs: a product photo on disk, a saved **model_response**, "
                "and **every** next-item ASIN (actual and suggested) must have a matching **.jpg** under "
                "**data/processed/image_cache/** so all tiles can show images."
            )
        else:
            def _example_label(i: int) -> str:
                title = examples[i].get("title") or "Product"
                tail = "…" if len(title) > 76 else ""
                return f"{i + 1}. {title[:76]}{tail}"

            labels = [_example_label(i) for i in range(len(examples))]
            idx = st.selectbox(
                "Pick an example",
                range(len(examples)),
                format_func=lambda i: labels[i],
            )
            ex = examples[idx]

            with st.container(border=True):
                img_c, text_c = st.columns([1, 2], gap="large")
                with img_c:
                    st.caption("They viewed this")
                    from src.demo.thumbnails import show_uniform_product_image

                    local_p = _anchor_image_path(ex)
                    show_uniform_product_image(
                        st,
                        local_path=local_p,
                        image_url=ex.get("image_url") if local_p is None else None,
                    )
                    st.metric("Rating", f"{ex.get('rating', '—')} ★" if ex.get("rating") is not None else "—")
                with text_c:
                    st.subheader(ex.get("title") or "Product")
                    if ex.get("main_category"):
                        st.caption(ex["main_category"])
                    if ex.get("review_title"):
                        st.caption(f'Customer review: "{ex["review_title"]}"')

            st.markdown('<div class="demo-compare-wrap"></div>', unsafe_allow_html=True)
            from src.demo.item_cards import (
                build_item_cards,
                render_horizontal_comparison_streamlit,
            )

            gt = ex.get("output") or ""
            cached = (ex.get("model_response") or "").strip()
            gt_cards = build_item_cards(gt, ex.get("output_names"), catalog, REPO_ROOT)

            live_s = (st.session_state.get(f"live_{idx}") or "").strip()
            pred_text = live_s if live_s else cached
            pred_cards = (
                build_item_cards(pred_text, None, catalog, REPO_ROOT) if pred_text else []
            )

            right_title = (
                "What the model suggests (just generated)"
                if live_s
                else "What the model suggested"
            )
            render_horizontal_comparison_streamlit(
                gt_cards,
                pred_cards,
                left_heading="What they bought next",
                right_heading=right_title,
            )

            bundle = st.session_state.get("bundle")
            if bundle and st.button(
                "Get new suggestions from the model", key=f"regen_{idx}", use_container_width=True
            ):
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
            '<p class="demo-lead">Describe a shopper and product the same way as in the examples, then get suggestions.</p>',
            unsafe_allow_html=True,
        )
        default_instr = (
            examples[0].get("instruction", "")
            if examples
            else "Given a user purchased an item with the following details, predict the next 5 items the user would purchase. "
        )
        instr = st.text_area("Instructions for the model", value=default_instr, height=88)
        user_input = st.text_area("Shopper and product", value=examples[0]["input"] if examples else "", height=140)
        bundle = st.session_state.get("bundle")
        go = st.button("Get suggestions", disabled=bundle is None, use_container_width=True)
        if go and bundle:
            model, tokenizer, special_chars, end_id, base_cfg, config, _ok = bundle
            from src.demo.inference import generate_recommendations
            from src.demo.item_cards import build_item_cards, render_item_cards_streamlit

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
                        "Suggested next products",
                    )
                except Exception as e:
                    st.error(str(e))
        elif bundle is None:
            st.info("Load the model in the sidebar first.")

    st.divider()
    st.caption("Recommendation research demo.")


if __name__ == "__main__":
    main()
