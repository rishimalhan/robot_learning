#! /usr/bin/env python3

from neural_engine.common import (
    load_model,
    load_tokenizer,
)
import torch
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load tokenizer
logging.info("Loading tokenizer...")
tokenizer = load_tokenizer()

# Boolean flag to switch between original and fine-tuned model
use_fine_tuned_model = True

# Load model
logging.info(f"Loading {'fine-tuned' if use_fine_tuned_model else 'original'} model...")
model = load_model(inference=use_fine_tuned_model)

# Prepare evaluation prompt
logging.info("Preparing evaluation prompt...")

# Set model to evaluation mode
model.eval()

# Perform inference
logging.info("Ready to perform inference...")


def generate_response(prompt, model, tokenizer):
    with torch.no_grad():
        model_input = tokenizer(prompt, return_tensors="pt").to("mps")
        output = model.generate(
            **model_input, max_new_tokens=128, repetition_penalty=1.15
        )
        logging.info("Inference complete. Decoding output...")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def evaluate_model(
    model, tokenizer, eval_data, num_samples=100, batch_size=4, max_new_tokens=128
):
    device = model.device
    results = []

    # Step 1: Prepare prompts and references
    eval_data_list = eval_data.to_list()[:num_samples]
    prompts = [item["text"][:20].strip() for item in eval_data_list]
    references = [item["text"][20:].strip() for item in eval_data_list]

    # Step 2: Iterate in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_references = references[i : i + batch_size]

        # Step 3: Tokenize the batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        # Step 4: Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Step 5: Decode predictions
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Step 6: Extract predictions (strip prompt from output)
        for prompt, reference, full_output in zip(
            batch_prompts, batch_references, decoded_outputs
        ):
            prediction = full_output.split("### Assistant:")[-1].strip()
            results.append(
                {"prompt": prompt, "reference": reference, "prediction": prediction}
            )

    return results


def compute_bleu(results):
    smoothie = SmoothingFunction().method4
    scores = []

    for row in results:
        reference = row["reference"].strip().split()
        prediction = row["prediction"].strip().split()
        score = sentence_bleu([reference], prediction, smoothing_function=smoothie)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"Average BLEU score: {avg_score:.4f}")
    return avg_score


def compute_bertscore(results, lang="en", model_type="microsoft/deberta-xlarge-mnli"):
    references = [row["reference"].strip() for row in results]
    predictions = [row["prediction"].strip() for row in results]

    print("Running BERTScore evaluation...")

    P, R, F1 = bert_score(predictions, references, lang=lang, model_type=model_type)

    avg_f1 = F1.mean().item()
    print(f"Average BERTScore F1: {avg_f1:.4f}")
    return avg_f1


# Loop to continuously prompt for input
while True:
    # Capture user input
    user_prompt = input("Enter your prompt (type 'exit' to quit): ")

    # Check if the user wants to exit
    if user_prompt.lower() == "exit":
        logging.info("Exit command received. Exiting...")
        break

    # Generate response based on user input
    generate_response(user_prompt, model, tokenizer)


# if __name__ == "__main__":
#     eval_data = load_and_prepare_dataset()
#     results = evaluate_model(model, tokenizer, eval_data, num_samples=10)
#     compute_bertscore(results)
#     from IPython import embed

#     embed()
