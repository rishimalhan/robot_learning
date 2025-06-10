#! /usr/bin/env python3

"""
PEFT: Parameter-Efficient Fine-Tuning
LoRA: Low rank Adaptation
Prompt engineering -> Q&A -> RAG -> Fine-tuning

What model 
needs to know
|
|   RAG                 All of above
|
|   Prompt              Finetuning
|   engineering
|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    How model needs to act

LoRa introduces an adapter with sparse matrix parameter to fine tune the
pre-trained model. The fundamental idea is that a pre-trained model has sparse weights
that are not changed during training. There are several adapters for different tasks.
"""

import os
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from neural_engine.common import (
    load_model,
    load_tokenizer,
    DATASET_PATH,
    OUTPUT_DIR,
    load_and_prepare_dataset,
)
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
MAX_LENGTH = 256
BATCH_SIZE = 8  # Adjusted batch size
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5

logging.info(f"Dataset path: {DATASET_PATH}")
logging.info(f"Output directory: {OUTPUT_DIR}")


def log_token_length_stats(dataset, tokenizer):
    """Log the minimum and maximum token lengths in the dataset."""
    logging.info("Calculating token length statistics...")
    token_lengths = []

    for data in dataset:
        tokens = tokenizer(data["text"])["input_ids"]
        token_lengths.append(len(tokens))

    min_length = min(token_lengths)
    max_length = max(token_lengths)

    logging.info(f"Minimum token length: {min_length}")
    logging.info(f"Maximum token length: {max_length}")


def main():
    # Load and prepare dataset
    dataset = load_and_prepare_dataset()
    tokenizer = load_tokenizer()
    # Tokenize dataset
    logging.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    log_token_length_stats(dataset, tokenizer)

    # Load model and tokenizer
    model = load_model(inference=False)
    model.train()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.print_trainable_parameters()

    # Prepare training arguments with MPS-specific settings
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=25,
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # Enable mixed precision
        bf16=True,
        gradient_accumulation_steps=4,  # Adjusted for efficiency
        optim="adamw_torch",  # Use PyTorch's AdamW optimizer
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
        warmup_steps=2,  # Add warmup
        gradient_checkpointing=True,  # Enable gradient checkpointing
        eval_steps=None,
        report_to="wandb",  # Enable wandb reporting
        run_name="mistral_fine_tuning",  # Set a specific run name
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    logging.info("Starting training...")
    trainer.train()

    # Save the model
    logging.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    logging.info(f"Training complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
