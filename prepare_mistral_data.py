#!/usr/bin/env python3
"""
Convert ForexGPT labeled training data to Mistral-Instruct format
for fine-tuning with Hugging Face
"""

import json
from pathlib import Path

def convert_to_mistral_format(labeled_data_path, output_path):
    """
    Convert labeled examples to Mistral-Instruct chat format
    
    Args:
        labeled_data_path: Path to training_data.json
        output_path: Path to save converted data
    """
    
    # Load labeled data
    print(f"Loading labeled data from {labeled_data_path}...")
    with open(labeled_data_path, 'r') as f:
        labeled_data = json.load(f)
    
    print(f"Found {len(labeled_data)} labeled examples")
    
    # System prompt for the model
    SYSTEM_PROMPT = """You are ForexGPT, a specialized AI assistant for extracting forex trading signals from corporate earnings call transcripts.

When given an earnings transcript excerpt, you analyze it for currency exposure information and return a structured JSON response with the following fields:
- signal (boolean): Whether a valid forex signal exists
- currency_pair (string): The currency pair (e.g., "EUR/USD")
- direction (string): Trading direction - "LONG", "SHORT", or "NEUTRAL"
- confidence (float): Confidence score between 0.0 and 1.0
- reasoning (string): Clear explanation of why this signal exists
- magnitude (string): Impact level - "low", "moderate", or "high"
- time_horizon (string): Expected timeframe - "next_week", "next_month", "next_quarter", or "next_year"

Only extract signals when there is clear forex exposure mentioned. Be conservative with confidence scores."""

    # Convert to Mistral format
    mistral_data = []
    
    for idx, example in enumerate(labeled_data):
        # User message with the transcript excerpt
        user_content = f"""Extract forex trading signals from this earnings call transcript. Return a structured JSON response.

Transcript:
{example['input']}"""
        
        # Assistant response with the extracted signal
        assistant_content = json.dumps(example['output'], ensure_ascii=False)
        
        # Create conversation in Mistral format
        mistral_example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        mistral_data.append(mistral_example)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Converted {idx + 1}/{len(labeled_data)} examples...")
    
    # Save converted data
    print(f"Saving converted data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(mistral_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully converted {len(mistral_data)} examples to Mistral format")
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE CONVERTED EXAMPLE:")
    print("="*60)
    print(json.dumps(mistral_data[0], indent=2, ensure_ascii=False)[:500] + "...")
    
    return mistral_data


def create_train_val_split(mistral_data, train_output, val_output, val_ratio=0.2):
    """
    Split data into training and validation sets
    
    Args:
        mistral_data: List of converted examples
        train_output: Path for training data
        val_output: Path for validation data
        val_ratio: Validation set ratio (default 0.2 = 80/20 split)
    """
    import random
    
    # Shuffle data
    random.seed(42)
    shuffled_data = mistral_data.copy()
    random.shuffle(shuffled_data)
    
    # Split
    split_idx = int(len(shuffled_data) * (1 - val_ratio))
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    print(f"\nSplitting data:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    
    # Save splits
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_output, 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved training data to {train_output}")
    print(f"✓ Saved validation data to {val_output}")


if __name__ == "__main__":
    # Paths
    labeled_data_path = "data/labeled/training_data.json"
    mistral_output_path = "data/labeled/mistral_format.json"
    train_output_path = "data/labeled/train.json"
    val_output_path = "data/labeled/val.json"
    
    # Create output directory if needed
    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    
    # Convert to Mistral format
    mistral_data = convert_to_mistral_format(labeled_data_path, mistral_output_path)
    
    # Create train/val split
    create_train_val_split(mistral_data, train_output_path, val_output_path)
    
    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Next steps:")
    print(f"1. Review the converted data in {mistral_output_path}")
    print(f"2. Use {train_output_path} and {val_output_path} for fine-tuning")
    print(f"3. Run the fine-tuning script with these files")
