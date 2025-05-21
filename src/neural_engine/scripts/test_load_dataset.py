#! /usr/bin/env python3

import kagglehub
import os
import pandas as pd
import json
import random
from pathlib import Path

# Flags to control data operations
DOWNLOAD_DATA = False
READ_DATA = True
CREATE_JSON = True

# Get the current script's directory and navigate to root folder
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets to scripts/
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(current_dir))
)  # Goes up three levels
data_dir = os.path.join(root_dir, "data")

if DOWNLOAD_DATA:
    # Download latest version
    path = kagglehub.dataset_download(
        # handle="shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes",
        handle="jithinanievarghese/drugs-side-effects-and-medical-condition",
        path=data_dir,
    )
    print("Path to dataset files:", path)

if READ_DATA:
    # Read the dataset from CSV
    csv_path = Path(
        os.path.join(data_dir, "drug_discovery", "drugs_side_effects_drugs_com.csv")
    )  # Replace with your path
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded dataset with {len(df)} rows")

        if CREATE_JSON:
            # Create messages for training
            messages = []

            for _, row in df.iterrows():
                condition = str(row["medical_condition"]).strip()
                drug = str(row["drug_name"]).strip()
                generic = str(row["generic_name"]).strip()
                brands = str(row["brand_names"]).strip()
                side_effects = str(row["side_effects"]).strip()
                description = str(row["medical_condition_description"]).strip()

                if not condition or not drug:
                    continue

                message = [f"{condition} is treated with {drug}."]
                if generic and generic.lower() != drug.lower():
                    message.append(f" (generic: {generic}).")
                if brands and brands.lower() != "nan":
                    message.append(f"Available brand names: {brands}.")
                message = " ".join(message)
                # Append to messages list
                messages.append(message)
                if description:
                    messages.append(f"{drug} is commonly used for: {description}.")
                if side_effects and side_effects.lower() != "nan":
                    messages.append(
                        f"Possible side effects of {drug} include: {side_effects}."
                    )

            # Wrap in final JSON structure
            final_output = {"messages": messages}

            # Save to JSON
            output_path = os.path.join(
                data_dir, "drug_discovery", "pharma_chat_dataset.json"
            )  # Output path
            with open(output_path, "w") as f:
                json.dump(final_output, f, indent=2)

            print(f"Generated {len(messages)} user-assistant pairs.")
            print(f"Saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find dataset at {csv_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
