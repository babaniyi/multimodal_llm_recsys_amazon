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


def _gallery_file_fingerprint() -> Tuple[float, float]:
    proc = REPO_ROOT / "data" / "processed"
    jl = proc / "processed_data.jsonl"
    tr = proc / "test_with_responses.json"
    return (
        jl.stat().st_mtime if jl.is_file() else 0.0,
        tr.stat().st_mtime if tr.is_file() else 0.0,
    )


@st.cache_data(show_spinner="Loading shopping examples…")
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

    st.title("Product recommendations")
    st.markdown(
        """
Pick a **real shopping moment** (we only show products that already have a photo on disk).  
See **what the customer bought next** in the data, compare to **our model’s suggestions**, or run the model live after loading it in the sidebar.
        """.strip()
    )

    catalog = _title_catalog()
    jl_mt, test_mt = _gallery_file_fingerprint()
    examples = _cached_gallery_examples(
        str(REPO_ROOT), jl_mt, test_mt, max_examples=36, seed=42
    )

    with st.sidebar:
        st.header("Recommendation model")
        st.caption(
            "After training, your model file is loaded here so you can generate fresh suggestions."
        )
        default_ckpt = REPO_ROOT / "output" / "model" / "finetuned-gpt2-medium-sft.pth"
        ckpt_input = st.text_input(
            "Trained model file",
            value=str(default_ckpt),
            help="Usually output/model/finetuned-gpt2-medium-sft.pth after training.",
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
            st.success("Loaded your trained model automatically.")
        err = st.session_state.pop("_demo_auto_load_error", None)
        if err:
            st.warning(f"Auto-load failed (use the button to retry): {err}")

        if st.button("Load / reload recommendation model"):
            try:
                bundle = _load_inference_bundle(ckpt_input, _checkpoint_mtime(ckpt_input))
                st.session_state["bundle"] = bundle
                st.session_state.pop("_demo_auto_fail_path", None)
                _, _, _, _, _, _, ok = bundle
                if ok:
                    st.success("Trained model loaded.")
                else:
                    st.warning(
                        "Trained model file not found — running a generic base model only, "
                        "so suggestions won’t match your trained system until you add the file."
                    )
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.session_state.pop("bundle", None)

        st.divider()
        if st.button("Reload shopping examples from disk"):
            _cached_gallery_examples.clear()
            st.rerun()

        st.caption(
            f"**{len(examples)}** shopping moments with photos · **{len(catalog):,}** product names in the catalog"
        )
        with st.expander("Technical details"):
            st.caption(f"Data folder: `{REPO_ROOT / 'data' / 'processed'}`")

    tab_gallery, tab_custom = st.tabs(["Examples", "Try your own"])

    with tab_gallery:
        if not examples:
            st.error(
                "No examples with a saved product image were found. "
                "Ensure `data/processed/processed_data.jsonl` lists `local_image_path` and those image files exist under `data/processed/image_cache/`."
            )
        else:
            def _example_label(i: int) -> str:
                title = examples[i].get("title") or "Product"
                tail = "…" if len(title) > 72 else ""
                return f"{i + 1}. {title[:72]}{tail}"

            labels = [_example_label(i) for i in range(len(examples))]
            idx = st.selectbox("Choose an example", range(len(examples)), format_func=lambda i: labels[i])
            ex = examples[idx]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.caption("Product they’re looking at")
                local_p = _anchor_image_path(ex)
                if local_p is not None:
                    st.image(str(local_p), use_container_width=True)
                elif ex.get("image_url"):
                    st.image(ex["image_url"], use_container_width=True)
                else:
                    st.info("No image for this product.")
                st.metric("Star rating", ex.get("rating", "—"))
                st.caption(f"**{ex.get('main_category', '—')}**")
            with c2:
                st.subheader(ex.get("title") or "Product")
                if ex.get("review_title"):
                    st.caption(f"What they said: *{ex['review_title']}*")
                with st.expander("Details shown to the model (for developers)"):
                    st.code(
                        (ex.get("instruction") or "") + "\n" + (ex.get("input") or ""),
                        language="text",
                    )

            st.divider()
            from src.demo.catalog import parse_prediction_tokens
            from src.demo.item_cards import build_item_cards, render_item_cards_streamlit

            gt = ex.get("output") or ""
            cached = (ex.get("model_response") or "").strip()
            gt_cards = build_item_cards(gt, ex.get("output_names"), catalog, REPO_ROOT)

            g1, g2 = st.columns(2)
            with g1:
                render_item_cards_streamlit(
                    gt_cards,
                    "What they bought next (from purchase history)",
                )
                with st.expander("For developers: internal product codes"):
                    _token_table(
                        _resolve_rows(parse_prediction_tokens(gt), catalog),
                        "Codes",
                    )
            with g2:
                live_s = (st.session_state.get(f"live_{idx}") or "").strip()
                if live_s:
                    render_item_cards_streamlit(
                        build_item_cards(live_s, None, catalog, REPO_ROOT),
                        "Our model’s suggestions (just generated)",
                    )
                    with st.expander("For developers: internal product codes"):
                        _token_table(
                            _resolve_rows(parse_prediction_tokens(live_s), catalog),
                            "Codes",
                        )
                elif cached:
                    render_item_cards_streamlit(
                        build_item_cards(cached, None, catalog, REPO_ROOT),
                        "Our model’s suggestions (from an earlier run)",
                    )
                    with st.expander("For developers: internal product codes"):
                        _token_table(
                            _resolve_rows(parse_prediction_tokens(cached), catalog),
                            "Codes",
                        )
                else:
                    st.markdown("##### Our model’s suggestions")
                    st.info(
                        "There’s no saved suggestion for this example. "
                        "Load the recommendation model in the sidebar, then click **Get new suggestions** below."
                    )

            bundle = st.session_state.get("bundle")
            if bundle and st.button("Get new suggestions", key=f"regen_{idx}"):
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

    with tab_custom:
        st.markdown(
            "Advanced: paste the **scenario text** your model expects (same format as in the examples — product and shopper details)."
        )
        default_instr = (
            examples[0].get("instruction", "")
            if examples
            else "Given a user purchased an item with the following details, predict the next 5 items the user would purchase. "
        )
        instr = st.text_area("Task wording", value=default_instr, height=100)
        user_input = st.text_area(
            "Scenario (shopper + product)",
            value=examples[0]["input"] if examples else "",
            height=160,
        )
        bundle = st.session_state.get("bundle")
        if st.button("Get suggestions", disabled=bundle is None):
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
                    from src.demo.item_cards import build_item_cards, render_item_cards_streamlit

                    render_item_cards_streamlit(
                        build_item_cards(out, None, catalog, REPO_ROOT),
                        "Suggested next products",
                    )
                    with st.expander("For developers: raw model output"):
                        st.code(out, language="text")
                    _token_table(_resolve_rows(parse_prediction_tokens(out), catalog), "Codes")
                except Exception as e:
                    st.error(str(e))
        elif bundle is None:
            st.info("Load the recommendation model in the sidebar first, then you can get suggestions here.")

    st.divider()
    st.caption("Research demo — MM-GPT2Rec (multimodal product recommendations).")


if __name__ == "__main__":
    main()
