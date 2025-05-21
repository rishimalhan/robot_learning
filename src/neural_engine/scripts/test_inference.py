from common import load_model, load_tokenizer
import torch
import logging

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


def generate_response(prompt):
    with torch.no_grad():
        model_input = tokenizer(prompt, return_tensors="pt").to("mps")
        output = model.generate(
            **model_input, max_new_tokens=128, repetition_penalty=1.15
        )
        logging.info("Inference complete. Decoding output...")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


# Loop to continuously prompt for input
while True:
    # Capture user input
    user_prompt = input("Enter your prompt (type 'exit' to quit): ")

    # Check if the user wants to exit
    if user_prompt.lower() == "exit":
        logging.info("Exit command received. Exiting...")
        break

    # Generate response based on user input
    generate_response(user_prompt)
