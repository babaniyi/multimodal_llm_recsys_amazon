"""Load MM-GPT2Rec and run the same generation path as training."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import tiktoken
import torch
from transformers.models.gpt2 import GPT2Model

from src.data.processor import format_input_phi
from src.models import model as model_fn
from src.models.model import load_weights
from src.utils.utils import get_device, load_config

logger = logging.getLogger(__name__)

MODEL_NAMES = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}


def _build_tokenizer(
    processed_dir: Path, special_user_item_ids: Set[str]
) -> Tuple[Any, int]:
    base_tokenizer = tiktoken.get_encoding("gpt2")
    custom_token_ids = {
        token: base_tokenizer.n_vocab + i
        for i, token in enumerate(special_user_item_ids)
    }
    tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
    )
    end_of_text_id = int(
        np.array(
            tokenizer.encode("<|endoftext|>", allowed_special=special_user_item_ids)
        ).item()
    )
    return tokenizer, end_of_text_id


def _extend_embeddings_and_head(model, device: str, n_special: int):
    num_tokens, emb_size = model.tok_emb.weight.shape
    new_embedding = torch.nn.Embedding(num_tokens + n_special, emb_size).to(device)
    new_embedding.weight.data[:num_tokens] = model.tok_emb.weight.data
    model.tok_emb = new_embedding

    original_out_features, original_in_features = model.out_head.weight.shape
    new_linear = torch.nn.Linear(original_in_features, original_out_features + n_special).to(
        device
    )
    with torch.no_grad():
        new_linear.weight[:original_out_features] = model.out_head.weight
        if model.out_head.bias is not None:
            new_linear.bias[:original_out_features] = model.out_head.bias
    model.out_head = new_linear


def build_model_for_demo(
    repo_root: Path,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any, Set[str], int, Dict[str, Any], Dict[str, Any]]:
    """Construct model + tokenizer; load GPT-2 weights from Hugging Face (no fine-tuned checkpoint)."""
    if config is None:
        config = load_config(str(repo_root / "configs" / "train_config.json"))
    processed_dir = repo_root / "data" / "processed"
    with open(processed_dir / "special_user_item_ids.json", encoding="utf-8") as f:
        special_list = json.load(f)
    special_user_item_ids: Set[str] = set(["<|endoftext|>"] + special_list)

    tokenizer, end_of_text_id = _build_tokenizer(processed_dir, special_user_item_ids)

    BASE_CONFIG = {
        "vocab_size": config["model_config"]["vocab_size"],
        "context_length": config["model_config"]["context_length"],
        "drop_rate": config["model_config"]["drop_rate"],
        "qkv_bias": config["model_config"]["qkv_bias"],
    }
    CHOOSE_MODEL = config["model_config"]["model_name"]
    BASE_CONFIG.update(config["gpt_model_config"][CHOOSE_MODEL])

    dev = device or str(get_device())
    cache_dir = str(repo_root / "checkpoints")
    model_hf = GPT2Model.from_pretrained(MODEL_NAMES[CHOOSE_MODEL], cache_dir=cache_dir)
    model_hf.eval()

    model = model_fn.GPTModel(BASE_CONFIG)
    load_weights(model, model_hf, BASE_CONFIG)
    model.to(dev)

    n_special = len(special_user_item_ids)
    _extend_embeddings_and_head(model, dev, n_special)

    if config["custom_config"].get("lora", False):
        for param in model.parameters():
            param.requires_grad = False
        # Match src.training.train (hardcoded there, not read from config)
        model_fn.replace_linear_with_lora(model, rank=16, alpha=16)
        model.to(dev)

    model.eval()
    return model, tokenizer, special_user_item_ids, end_of_text_id, BASE_CONFIG, config


def load_fine_tuned_weights(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Checkpoint missing keys (sample): %s", missing[:5])
    if unexpected:
        logger.warning("Checkpoint unexpected keys (sample): %s", unexpected[:5])


def generate_recommendations(
    model: torch.nn.Module,
    tokenizer: Any,
    special_chars: Set[str],
    end_of_text_id: int,
    base_config: Dict[str, Any],
    entry: Dict[str, Any],
    training_config: Dict[str, Any],
    device: str,
) -> str:
    """Phi-3-style prompt from `entry`, then sample tokens (same as train.py)."""
    input_text = format_input_phi(entry)
    input_ids = model_fn.text_to_token_ids(input_text, tokenizer, special_chars).to(device)
    with torch.no_grad():
        token_ids = model_fn.generate(
            model=model,
            idx=input_ids,
            max_new_tokens=int(training_config.get("max_new_tokens", 150)),
            context_size=base_config["context_length"],
            eos_id=end_of_text_id,
            temperature=float(training_config.get("temperature", 0.7)),
            top_k=int(training_config.get("top_k", 50)),
            top_p=float(training_config.get("top_p", 0.9)),
        )
    generated_text = model_fn.token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text) :].split("<|assistant|>:")[-1].strip()
    asin_pattern = r"<\|ASIN_[^|]+\|>"
    endoftext_pattern = r"<\|endoftext\|>"
    matches = re.findall(f"({asin_pattern}|{endoftext_pattern})", response_text)
    return ", ".join(matches)


def default_checkpoint_path(repo_root: Path, model_name: str) -> Path:
    safe = re.sub(r"[ ()]", "", model_name)
    return repo_root / "output" / "model" / f"finetuned-{safe}-sft.pth"
