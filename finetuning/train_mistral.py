#!/usr/bin/env python3
"""
Fine-tune Mistral-7B-Instruct-v0.3 for ForexGPT Signal Extraction
Using LoRA (Low-Rank Adaptation) for efficient training
"""

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "models/forexgpt-mistral-7b-lora"
TRAIN_DATA_PATH = "data/labeled/train.json"
VAL_DATA_PATH = "data/labeled/val.json"

# LoRA Configuration (as discussed)
LORA_CONFIG = {
    "r": 8,                    # Rank
    "lora_alpha": 16,          # Alpha
    "target_modules": [        # Which layers to apply LoRA to
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "max_seq_length": 2048,
    "warmup_steps": 10,
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "fp16": True,  # Use FP16 for faster training
}


def format_chat_template(example, tokenizer):
    """
    Format messages into Mistral's chat template
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def load_and_prepare_data(train_path, val_path, tokenizer):
    """
    Load training and validation data and format for training
    """
    print("Loading datasets...")
    
    # Load JSON data
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Format with chat template
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer():
    """
    Load model and tokenizer with 4-bit quantization for efficient training
    """
    print(f"Loading model: {MODEL_NAME}...")
    
    # 4-bit quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def train():
    """
    Main training function
    """
    print("="*60)
    print("FOREXGPT FINE-TUNING SCRIPT")
    print("Model: Mistral-7B-Instruct-v0.3")
    print("Method: LoRA (Low-Rank Adaptation)")
    print("="*60)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare datasets
    train_dataset, val_dataset = load_and_prepare_data(
        TRAIN_DATA_PATH, 
        VAL_DATA_PATH, 
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
        report_to="none",  # Set to "wandb" if you want to use Weights & Biases
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        packing=False,  # Don't pack multiple examples together
    )
    
    # Start training
    print("\nStarting training...")
    print("="*60)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*60)
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("="*60)
    
    return trainer


def test_model(model_path=OUTPUT_DIR):
    """
    Quick test of the fine-tuned model
    """
    print("\nTesting fine-tuned model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Test prompt
    test_transcript = """In Q1, we experienced a 4% revenue headwind from currency movements, 
    primarily due to USD strength versus EUR. Our European operations represent approximately 
    35% of total revenue, and we expect this headwind to continue into Q2."""
    
    messages = [
        {"role": "user", "content": f"Extract forex trading signals from this earnings call transcript. Return a structured JSON response.\n\nTranscript:\n{test_transcript}"}
    ]
    
    # Generate
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = pipe(
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    print("\nTest Response:")
    print("="*60)
    print(response[0]["generated_text"][-1]["content"])
    print("="*60)


if __name__ == "__main__":
    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable not set")
        print("You may need to authenticate with Hugging Face")
        print("Run: huggingface-cli login")
    
    # Run training
    trainer = train()
    
    # Optional: Test the model
    print("\nWould you like to test the model? (This will load the model again)")
    # Uncomment to enable testing:
    # test_model()
