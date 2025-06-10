import os
import torch
import logging
from datasets import Dataset
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants for paths and directories
PARENT_DIR = "/Users/rishimalhan/Software/robot_learning"
DATA_DIR = os.path.join(PARENT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Device configuration
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


def load_and_prepare_dataset(dataset_path: str) -> Dataset:
    """Load a JSON dataset and convert it to HuggingFace Dataset format."""
    logging.info(f"Loading dataset from {dataset_path}...")

    # Load JSON data
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Convert user-assistant pairs into structured training samples
    formatted_data = []
    for item in data["messages"]:
        formatted_data.append({"text": item})

    logging.info(f"Prepared {len(formatted_data)} training samples from dataset.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(formatted_data)
    return dataset
