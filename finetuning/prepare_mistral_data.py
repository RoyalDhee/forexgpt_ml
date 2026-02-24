#!/usr/bin/env python3
"""
Convert ForexGPT labeled training data to Mistral-Instruct format
with PROPER 3-WAY SPLIT: Train / Validation / Test
"""

import json
from pathlib import Path
import random

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
- time_horizon (string): Expected timeframe - "current_quater", "long_term", "next_quarter", or "null"

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
    
    print(f"Successfully converted {len(mistral_data)} examples to Mistral format")
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE CONVERTED EXAMPLE:")
    print("="*60)
    print(json.dumps(mistral_data[0], indent=2, ensure_ascii=False)[:500] + "...")
    
    return mistral_data


def create_three_way_split(mistral_data, train_output, val_output, test_output, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into training and validation sets
    
    Args:
        mistral_data: List of converted examples
        train_output: Path for training data
        val_output: Path for validation data
        test_output: Path for test data
        train_ratio: Training set ratio (default 0.70)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
    """
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"

    # Shuffle data
    random.seed(42)
    shuffled_data = mistral_data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total = len(shuffled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    print("\n" + "="*60)
    print("THREE-WAY DATA SPLIT:")
    print("="*60)
    print(f"Total examples: {total}")
    print(f"  Training examples:   {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation examples: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test examples:       {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    print("="*60)
    print("\nPurpose of each split:")
    print("  Training:   Used to train the model (update weights)")
    print("  Validation: Monitor during training, select best checkpoint")
    print("  Test:       Final evaluation AFTER training (unbiased metrics)")
    print("="*60)
    
    # Save splits
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_output, 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved training data to {train_output}")
    print(f"✓ Saved validation data to {val_output}")
    print(f"✓ Saved test data to {test_output}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Paths
    labeled_data_path = "data/labeled/training_data.json"
    mistral_output_path = "data/labeled/mistral_format.json"
    train_output_path = "data/labeled/train.json"
    val_output_path = "data/labeled/val.json"
    test_output_path = "data/labeled/test.json"  # NEW: Test set
    
    # Create output directory if needed
    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    
    # Convert to Mistral format
    mistral_data = convert_to_mistral_format(labeled_data_path, mistral_output_path)
    
    # Create train/val/test split (70/15/15)
    train_data, val_data, test_data = create_three_way_split(
        mistral_data, 
        train_output_path, 
        val_output_path,
        test_output_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Next steps:")
    print(f"1. Review the splits - ensure they look balanced")
    print(f"2. Use train.json and val.json for fine-tuning")
    print(f"3. ONLY use test.json AFTER training for final evaluation")
    print(f"4. Never look at test.json during training!")
    print("="*60)
