import torch
from functools import partial
from pathlib import Path
import json
import logging
import tiktoken
from torch.utils.data import DataLoader
import os
import urllib.request
import time
from tqdm import tqdm
import re

from ..data.processor import custom_collate_fn, format_input_phi, format_input_alpaca
from ..data.dataset import AmazonDatasetPhi, AmazonDatasetAlpaca
from ..utils.utils import get_device, load_config, setup_logging, set_seed
from ..models import model as model_fn
from ..utils.visualization import plot_cross_entropy_loss, plot_losses_over_tokens_seen


setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)


def main():

    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cache_dir = processed_dir / "image_cache"

    model_dir = Path("output") / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Potential error: These files might not exist
    try:
        end_of_text_id = json.load(open(processed_dir / "end_of_text_id.json"))
        special_user_item_ids = json.load(open(processed_dir / "special_user_item_ids.json"))
    except FileNotFoundError as e:
        logger.error(f"Required JSON file not found: {e}")
        raise

    # =================================================================================
    # 1.0 Set custom collate function
    # =================================================================================
    device = get_device()
    
    # Potential error: These files might not exist
    try:
        train_data = json.load(open(processed_dir / "train.json"))
        val_data = json.load(open(processed_dir / "val.json"))
        test_data = json.load(open(processed_dir / "test.json"))
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        raise

    logger.info("Finished loading train, val and test data")

    customized_collate_fn = partial(custom_collate_fn, pad_token_id=end_of_text_id, 
                                    allowed_max_length=config['model_config']['context_length'], 
                                    device=device)
    
    logger.info("Finished setting custom collate function")
    phi3_prompt = config['custom_config']['phi3']
    alpaca_prompt = config['custom_config']['alpaca']

    if phi3_prompt:
        CustomDataset = AmazonDatasetPhi
    else:
        CustomDataset = AmazonDatasetAlpaca

    # 1.1 Set the data loaders
    #_______________________________________________________________________________
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    train_dataset = CustomDataset(train_data, tokenizer, special_user_item_ids)
    val_dataset = CustomDataset(val_data, tokenizer, special_user_item_ids)

    train_loader = DataLoader(
        train_dataset, batch_size=config['model_config']['batch_size'], 
        collate_fn=customized_collate_fn, drop_last=True,
        shuffle=False, num_workers=config['model_config']['num_workers'])
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['model_config']['batch_size'], 
        collate_fn=customized_collate_fn, drop_last=True,
        shuffle=False, num_workers=config['model_config']['num_workers'])
    
    logger.info("Finished setting the data loaders")

    # 1.2 Load pretrained model
    #_______________________________________________________________________________
    BASE_CONFIG = {
        "vocab_size": config['model_config']['vocab_size'],     # Vocabulary size
        "context_length": config['model_config']['context_length'],  # Context length
        "drop_rate": config['model_config']['drop_rate'],        # Dropout rate
        "qkv_bias": config['model_config']['qkv_bias']         # Query-key-value bias
    }

    model_configs = config['gpt_model_config']
    CHOOSE_MODEL = config['model_config']['model_name']

    # Potential error: Model name might not be in configs
    if CHOOSE_MODEL not in model_configs:
        raise ValueError(f"Model {CHOOSE_MODEL} not found in model_configs")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    if CHOOSE_MODEL == "gpt2-small (124M)":
        filename = "gpt2-small-124M.pth"
    elif CHOOSE_MODEL == "gpt2-medium (355M)":
        filename = "gpt2-medium-355M.pth"
    elif CHOOSE_MODEL == "gpt2-large (774M)":
        filename = "gpt2-large-774M.pth"
    elif CHOOSE_MODEL == "gpt2-xl (1558M)":
        filename = "gpt2-xl-1558M.pth"
    else:
        raise ValueError(f"Invalid model name: {CHOOSE_MODEL}")

    # Fix typo in variable name
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{filename}"

    model_path = model_dir / filename
    if not os.path.exists(model_path):
        logger.info(f"Downloading model {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
        logger.info(f"Downloaded to {model_path}")
    else:
        logger.info(f"Model {filename} already exists at {model_path}")

    try:
        model = model_fn.GPTModel(BASE_CONFIG)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    logger.info(f"Finished loading the weights: {CHOOSE_MODEL}")

    lora = config['custom_config']['lora']

    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters after: {total_params:,}")
        model_fn.replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    # =================================================================================
    # 2: Start finetuning the model
    # =================================================================================
    logging.info("Starting finetuning the model")

    # Initial Loss
    #_______________________________________________________________________________
    logger.info("> Initial losses")
    with torch.no_grad():
        train_loss = model_fn.calc_loss_loader(train_loader, model, device, num_batches=4)
        val_loss = model_fn.calc_loss_loader(val_loader, model, device, num_batches=4)

    logger.info(f"   Training loss: {train_loss}")
    logger.info(f"   Validation loss: {val_loss}")
    logger.info(f"   Percentage change: {round(100 * (val_loss - train_loss)/train_loss, 2)}%")

    # Training
    #_______________________________________________________________________________
    start_time = time.time()

    num_epochs = config['training_config']['num_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training_config']['learning_rate'], weight_decay=config['training_config']['weight_decay'])

    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input_alpaca(val_data[0])

    train_losses, val_losses, tokens_seen = model_fn.train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer,
        special_chars=special_user_item_ids,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"> Training completed in {execution_time_minutes:.2f} minutes.")


    logger.info("Plotting losses")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_name = "loss-plot.pdf"
    if phi3_prompt:
        plot_name_ce = plot_name.replace(".pdf", "-phi3-prompt.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-phi3-prompt-tokens.pdf")
    if lora:
        plot_name_ce = plot_name.replace(".pdf", "-lora.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-lora-tokens.pdf")
    if not any([phi3_prompt, lora]):
        plot_name_ce = plot_name.replace(".pdf", "-baseline.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-baseline-tokens.pdf")

    plot_dir = Path("output") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path_ce = plot_dir / plot_name_ce
    plot_path_tokens = plot_dir / plot_name_tokens

    try:
        plot_cross_entropy_loss(train_losses, val_losses, save_path=plot_path_ce)
        plot_losses_over_tokens_seen(epochs_tensor, tokens_seen, train_losses, val_losses, save_path=plot_path_tokens)
        logger.info(f"> Saved plots to {plot_path_ce} and {plot_path_tokens}")
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")

    # =================================================================================
    # 3: Generate responses
    # =================================================================================
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating responses"):
        try:
            input_text = format_input_phi(entry) if phi3_prompt else format_input_alpaca(entry)

            token_ids = model_fn.generate(
                model=model,
                idx=model_fn.text_to_token_ids(input_text, tokenizer, special_user_item_ids).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=end_of_text_id,
            )
            generated_text = model_fn.token_ids_to_text(token_ids, tokenizer)

            if phi3_prompt:
                response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
            else:
                response_text = generated_text[len(input_text):].replace("### Response:", "").strip()   

            test_data[i]["model_response"] = response_text
        except Exception as e:
            logger.error(f"Failed to generate response for entry {i}: {e}")
            test_data[i]["model_response"] = ""

    # =================================================================================
    # 4: Save the model 
    # =================================================================================
    try:
        model_path = model_dir /  f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # =================================================================================
    # 5: Save the test data with responses
    # =================================================================================
    try:
        test_data_path = processed_dir / "test_with_responses.json"
        with open(test_data_path, "w") as file:
            json.dump(test_data, file, indent=4)
        logger.info(f"Saved test data with responses to {test_data_path}")
    except Exception as e:
        logger.error(f"Failed to save test data responses: {e}")


if __name__ == "__main__":
    main()