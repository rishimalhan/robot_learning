import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PARENT_DIR = "/Users/rishimalhan/Software/robot_learning"
DATA_DIR = os.path.join(PARENT_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "drug_discovery", "pharma_chat_dataset.json")
MODELS_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(MODELS_DIR, "fine_tuned_mistral")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        logging.warning(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        logging.warning(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    DEVICE = "cpu"
else:
    DEVICE = "mps"
    logging.info("Using MPS device")


def load_tokenizer() -> AutoTokenizer:
    """Load the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right", cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(inference: bool = False) -> AutoModelForCausalLM:
    """Load the model, optionally loading the fine-tuned version."""
    model_path = (
        os.path.join(DATA_DIR, "models", "fine_tuned_mistral")
        if inference
        else MODEL_NAME
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
        device_map="auto" if DEVICE == "cpu" else None,
        cache_dir=CACHE_DIR,
        use_cache=False,  # Disable KV cache for training
    )

    if not inference:
        # Configure LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Move model to MPS device if available
    if DEVICE == "mps":
        model = model.to(DEVICE)
        # Enable memory efficient attention if available
        if hasattr(model, "enable_memory_efficient_attention"):
            model.enable_memory_efficient_attention()

    # Set pad_token_id to eos_token_id if not already set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.train()

    return model
