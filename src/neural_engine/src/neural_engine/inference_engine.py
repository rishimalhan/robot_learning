#!/usr/bin/env python3

import torch
import logging
from typing import Optional, Iterator, Union
from dataclasses import dataclass
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model
from neural_engine.common import DEVICE, CACHE_DIR, MODELS_DIR

# Model Constants
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
FINE_TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "fine_tuned_mistral")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 150  # Slightly longer to allow complete thoughts
    temperature: float = 0.7  # More balanced temperature
    top_p: float = 0.9  # Standard top_p value
    repetition_penalty: float = 1.1  # Gentler repetition penalty
    do_sample: bool = True
    min_tokens: int = 32  # Increased minimum tokens
    repetition_threshold: int = 8  # More lenient repetition threshold
    min_unique_tokens: int = 4  # Require more unique tokens before stopping


class InferenceEngine:
    """Optimized PyTorch-based inference engine for language models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEVICE,
        dtype: torch.dtype = torch.float16,
        use_fine_tuned: bool = True,
        **kwargs,
    ):
        """Initialize the inference engine.

        Args:
            model_name: Name or path of the model to load
            device: Device to run inference on ('mps', 'cuda', 'cpu')
            dtype: Data type for model weights
            use_fine_tuned: Whether to use the fine-tuned model
            **kwargs: Additional arguments passed to model loading
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.dtype = dtype

        # Initialize model and tokenizer
        self.logger.info(
            f"Loading {'fine-tuned' if use_fine_tuned else 'base'} model..."
        )
        self.model = self._load_model(model_name, use_fine_tuned, **kwargs)
        self.tokenizer = self._load_tokenizer(model_name)

        # Initialize generation config
        self.default_config = GenerationConfig()

        # Initialize KV cache
        self.kv_cache = {}
        self.logger.info("Inference engine initialized successfully")

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load and configure the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left", cache_dir=CACHE_DIR
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(
        self, model_name: str, use_fine_tuned: bool = False, **kwargs
    ) -> PreTrainedModel:
        """Load the model, optionally loading the fine-tuned version."""
        model_path = FINE_TUNED_MODEL_PATH if use_fine_tuned else model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cpu" else None,
            cache_dir=CACHE_DIR,
            use_cache=True,  # Enable KV cache for inference
            **kwargs,
        )

        if not use_fine_tuned:
            # Configure LoRA for base model
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

        # Move model to device
        model = model.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(model, "enable_memory_efficient_attention"):
            model.enable_memory_efficient_attention()

        # Set pad_token_id to eos_token_id if not already set
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        # Set to evaluation mode
        model.eval()
        return model

    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """Tokenize and prepare input for the model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Set a reasonable max length
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_next_token_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[tuple] = None,
    ) -> tuple:
        """Get probability distribution for next token."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            return next_token_logits, past_key_values

    def _sample_next_token(
        self, logits: torch.Tensor, config: GenerationConfig, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sample next token based on logits and generation config."""
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, input_ids, config.repetition_penalty
            )

        # Temperature scaling
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-p sampling
        if config.do_sample:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs > config.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0

            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            probs.scatter_(1, sorted_indices, sorted_probs)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

        return next_token

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
        return logits

    def _check_repetition(self, last_tokens: list, config: GenerationConfig) -> bool:
        """Check if the generation is stuck in a repetition loop."""
        if len(last_tokens) > config.repetition_threshold:
            unique_tokens = len(set(last_tokens[-config.repetition_threshold :]))
            return unique_tokens < config.min_unique_tokens
        return False

    def _generate_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Core text generation logic used by both streaming and non-streaming modes."""
        config = config or self.default_config

        # Prepare input
        model_inputs = self._prepare_input(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        past_key_values = None

        # Initialize generation tracking
        generated_tokens = []
        all_token_ids = input_ids.clone()
        token_count = 0
        repetition_window = []

        while token_count < config.max_tokens:
            # Get next token probabilities
            logits, past_key_values = self._get_next_token_probs(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            # Sample next token
            next_token = self._sample_next_token(logits, config, all_token_ids)
            token_count += 1

            # Update tracking
            generated_tokens.append(next_token.item())
            repetition_window = (repetition_window + [next_token.item()])[
                -config.repetition_threshold :
            ]

            # Update model inputs
            input_ids = next_token.view(1, 1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((1, 1))], dim=-1
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=1)

            # Check stopping conditions
            if (
                next_token.item() == self.tokenizer.eos_token_id
                and len(generated_tokens) >= config.min_tokens
            ):
                break

            if len(repetition_window) == config.repetition_threshold:
                unique_tokens = len(set(repetition_window))
                if unique_tokens < config.min_unique_tokens:
                    break

        # Decode only the generated response (excluding prompt)
        response = self.tokenizer.decode(
            all_token_ids[0, model_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response.strip()

    def _generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate response with streaming."""
        config = config or self.default_config

        # Prepare input
        model_inputs = self._prepare_input(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        past_key_values = None

        # Initialize generation tracking
        generated_tokens = []
        all_token_ids = input_ids.clone()
        token_count = 0
        repetition_window = []
        last_output_len = 0

        while token_count < config.max_tokens:
            # Get next token probabilities
            logits, past_key_values = self._get_next_token_probs(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            # Sample next token
            next_token = self._sample_next_token(logits, config, all_token_ids)
            token_count += 1

            # Update tracking
            generated_tokens.append(next_token.item())
            repetition_window = (repetition_window + [next_token.item()])[
                -config.repetition_threshold :
            ]

            # Update model inputs
            input_ids = next_token.view(1, 1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((1, 1))], dim=-1
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=1)

            # Decode and yield new text
            current_output = self.tokenizer.decode(
                all_token_ids[0, model_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            if len(current_output) > last_output_len:
                yield current_output[last_output_len:]
                last_output_len = len(current_output)

            # Check stopping conditions
            if (
                next_token.item() == self.tokenizer.eos_token_id
                and len(generated_tokens) >= config.min_tokens
            ):
                break

            if len(repetition_window) == config.repetition_threshold:
                unique_tokens = len(set(repetition_window))
                if unique_tokens < config.min_unique_tokens:
                    break

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, Iterator[str]]:
        """Generate text based on the prompt.

        Args:
            prompt: Input prompt text
            stream: Whether to stream the response token by token
            config: Optional generation configuration

        Returns:
            If stream=False, returns complete response as string
            If stream=True, returns iterator of response tokens
        """
        if stream:
            return self._generate_stream(prompt, config)
        else:
            return self._generate_text(prompt, config)


def main():
    """Example usage of the inference engine."""
    logging.basicConfig(level=logging.INFO)

    # Initialize engine with fine-tuned model
    engine = InferenceEngine(use_fine_tuned=True)

    # Example non-streaming generation
    prompt = "I have acne, what should I do? Give available brand names."
    response = engine.generate(prompt, stream=False)
    print(f"\nNon-streaming response:\n{response}")

    # Example streaming generation
    print("\nStreaming response:")
    for token in engine.generate(prompt, stream=True):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
