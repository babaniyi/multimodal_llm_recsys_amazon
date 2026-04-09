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


def _load_demo_examples() -> List[Dict[str, Any]]:
    path = REPO_ROOT / "demo" / "demo_examples.json"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading product catalog…")
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


@st.cache_resource(show_spinner="Loading language model (first run may download weights)…")
def _load_inference_bundle(checkpoint_path_str: str, checkpoint_mtime: float):
    """checkpoint_mtime busts the cache when the file is replaced after retraining."""
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


def _resolve_rows(tokens: List[str], catalog: Dict[str, str]) -> List[Tuple[str, str]]:
    from src.demo.catalog import resolve_tokens

    return resolve_tokens(tokens, catalog)


def _token_table(rows: List[Tuple[str, str]], title: str) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.caption("No structured tokens found.")
        return
    for i, (tok, name) in enumerate(rows, start=1):
        st.markdown(f"{i}. `{name}`")
        if tok != name and tok != "<|endoftext|>":
            st.caption(tok)


def main():
    st.set_page_config(
        page_title="MM-GPT2Rec Demo",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("MM-GPT2Rec — multimodal product recommendation")
    st.markdown(
        """
This demo shows **MM-GPT2Rec**: a GPT-2–based model fine-tuned to predict **the next products**
a user may buy from **Amazon-style context** (user id, item id, rating, review title, product title, category).
Predictions use special tokens such as `<|ASIN_B0123456789|>` and may end with `<|endoftext|>` when the model
signals *no further purchase* in the horizon.
        """.strip()
    )

    catalog = _title_catalog()
    examples = _load_demo_examples()

    with st.sidebar:
        st.header("Live inference")
        st.caption(
            "Weights: **OpenAI GPT-2 Medium** (HF) → then your **fine-tuned** `.pth` on top. "
            "We do not retrain; the checkpoint holds LoRA + extended item embeddings from training."
        )
        default_ckpt = REPO_ROOT / "output" / "model" / "finetuned-gpt2-medium-sft.pth"
        ckpt_input = st.text_input(
            "Fine-tuned checkpoint path",
            value=str(default_ckpt),
            help="Saved by training as output/model/finetuned-gpt2-medium-sft.pth",
        )

        # Use your trained weights as soon as the file exists (no extra click).
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
            st.success("Loaded your fine-tuned checkpoint automatically.")
        err = st.session_state.pop("_demo_auto_load_error", None)
        if err:
            st.warning(f"Auto-load failed (use the button to retry): {err}")

        if st.button("Load / reload model"):
            try:
                bundle = _load_inference_bundle(ckpt_input, _checkpoint_mtime(ckpt_input))
                st.session_state["bundle"] = bundle
                st.session_state.pop("_demo_auto_fail_path", None)
                _, _, _, _, _, _, ok = bundle
                if ok:
                    st.success("Fine-tuned weights loaded.")
                else:
                    st.warning(
                        "Checkpoint not found — loaded **base** GPT-2 + extended embeddings only "
                        "(outputs will not match paper results until you train and save weights)."
                    )
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.session_state.pop("bundle", None)

        st.divider()
        st.caption(
            f"Repository: `{REPO_ROOT}` · Catalog: **{len(catalog):,}** titles · "
            f"Examples: **{len(examples)}**"
        )

    tab_gallery, tab_custom = st.tabs(["Example gallery", "Try your own prompt"])

    with tab_gallery:
        if not examples:
            st.error(
                "No `demo/demo_examples.json`. Run `python demo/build_demo_examples.py` from the repo root."
            )
        else:
            labels = [
                f"{i + 1}. {ex.get('title', 'Untitled')[:70]}{'…' if len(ex.get('title') or '') > 70 else ''}"
                for i, ex in enumerate(examples)
            ]
            idx = st.selectbox("Choose a scenario", range(len(examples)), format_func=lambda i: labels[i])
            ex = examples[idx]

            c1, c2 = st.columns([1, 2])
            with c1:
                if ex.get("image_url"):
                    st.image(ex["image_url"], caption="Product image (Amazon)", use_container_width=True)
                else:
                    st.info("No image URL for this row.")
                st.metric("Rating", ex.get("rating", "—"))
                st.caption(f"Category: **{ex.get('main_category', '—')}**")
            with c2:
                st.subheader("Current product")
                st.write(ex.get("title", ""))
                if ex.get("review_title"):
                    st.caption(f"Review title: *{ex['review_title']}*")
                with st.expander("Raw model input (instruction + user context)"):
                    st.code(
                        (ex.get("instruction") or "") + "\n" + (ex.get("input") or ""),
                        language="text",
                    )

            st.divider()
            gt = ex.get("output") or ""
            cached = ex.get("model_response") or ""
            g1, g2 = st.columns(2)
            with g1:
                from src.demo.catalog import parse_prediction_tokens

                _token_table(_resolve_rows(parse_prediction_tokens(gt), catalog), "Held-out next items (ground truth)")
                if ex.get("output_names"):
                    with st.expander("Ground-truth product names (dataset)"):
                        st.text(ex["output_names"])
            with g2:
                _token_table(
                    _resolve_rows(parse_prediction_tokens(cached), catalog),
                    "MM-GPT2Rec prediction (cached from evaluation run)",
                )

            bundle = st.session_state.get("bundle")
            if bundle and st.button("Regenerate with loaded model", key=f"regen_{idx}"):
                model, tokenizer, special_chars, end_id, base_cfg, config, _ok = bundle
                from src.demo.inference import generate_recommendations

                device = str(next(model.parameters()).device)
                entry = {
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                }
                with st.spinner("Generating…"):
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
                    except Exception as e:
                        st.error(str(e))

            if bundle and f"live_{idx}" in st.session_state:
                from src.demo.catalog import parse_prediction_tokens

                st.divider()
                _token_table(
                    _resolve_rows(parse_prediction_tokens(st.session_state[f"live_{idx}"]), catalog),
                    "Live model output (persists until you change example or reload)",
                )

    with tab_custom:
        st.markdown(
            "Paste the **user context** line (the same text that appears after the instruction in the dataset), "
            "including user and ASIN tokens."
        )
        default_instr = (
            examples[0].get("instruction", "")
            if examples
            else "Given a user purchased an item with the following details, predict the next 5 items the user would purchase. "
        )
        instr = st.text_area("Instruction", value=default_instr, height=100)
        user_input = st.text_area(
            "Input (user + item description)",
            value=examples[0]["input"] if examples else "",
            height=160,
        )
        bundle = st.session_state.get("bundle")
        if st.button("Run recommendation", disabled=bundle is None):
            model, tokenizer, special_chars, end_id, base_cfg, config, ok = bundle
            from src.demo.inference import generate_recommendations

            device = str(next(model.parameters()).device)
            entry = {"instruction": instr, "input": user_input}
            with st.spinner("Generating…"):
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
                    from src.demo.catalog import parse_prediction_tokens

                    _token_table(_resolve_rows(parse_prediction_tokens(out), catalog), "Model output")
                    st.code(out, language="text")
                except Exception as e:
                    st.error(str(e))
        elif bundle is None:
            st.info('Use the sidebar to **Load / reload model** before running a custom prompt.')

    st.divider()
    st.caption(
        "Paper: *Enhancing Product Recommendation with Multimodal LLMs* — evaluation metrics and baselines are in the main README."
    )


if __name__ == "__main__":
    main()
