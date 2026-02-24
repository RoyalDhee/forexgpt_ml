#!/usr/bin/env python3
"""
Fine-tune Mistral-7B-Instruct-v0.3 for ForexGPT Signal Extraction
Using LoRA (Low-Rank Adaptation) for efficient training
WITH WEIGHT & BIASES TRACKING
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

# Weights & Biases Configuration
WANDB_PROJECT = "forexgpt"
WANDB_RUN_NAME = "mistral-7b-lora-v1"   # This run's name
WANDB_ENTITY = None                     # Wandb username (None = use default)

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
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False
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

    # Log dataset info to wandb
    wandb.config.update({
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "total_example": len(train_dataset) + len(val_dataset)
    })
    
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
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {trainable_percent:.4f}%")
    
    # Log model config to wandb
    wandb.config.update({
        "model_name": MODEL_NAME,
        "lora_r": LORA_CONFIG["r"],
        "lora_alpha": LORA_CONFIG["lora_alpha"],
        "lora_dropout": LORA_CONFIG["lora_dropout"],
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
        "quantization": "4-bit NF4",
    })
    
    return model, tokenizer


def train():
    """
    Main training function with Weights & Biases tracking
    """
    print("="*60)
    print("FOREXGPT FINE-TUNING SCRIPT")
    print("Model: Mistral-7B-Instruct-v0.3")
    print("Method: LoRA (Low-Rank Adaptation)")
    print("Tracking: Weights & Biases")
    print("="*60)
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        entity=WANDB_ENTITY,
        config={
            **LORA_CONFIG,
            **TRAINING_CONFIG,
            "model_name": MODEL_NAME,
            "output_dir": OUTPUT_DIR,
        }
    )

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare datasets
    train_dataset, val_dataset = load_and_prepare_data(
        TRAIN_DATA_PATH, 
        VAL_DATA_PATH, 
        tokenizer
    )

    # Training arguments with wandb integration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
        report_to="wandb",  # Enable wandb logging
        run_name=WANDB_RUN_NAME,
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
    print(f"View training progress at: {wandb.run.get_url()}")
    print("="*60)

    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Log final metrics
    final_metrics = {
        "final_train_loss": trainer.state.log_history[-2]["loss"] if len(trainer.state.log_history) > 1 else None,
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", None),
    }
    wandb.log(final_metrics)
    
    print("="*60)
    print(f"Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"View full training report at: {wandb.run.get_url()}")
    print("="*60)

    # Finish wandb run
    wandb.finish()
    
    return trainer


def test_model(model_path=OUTPUT_DIR):
    """
    Quick test of the fine-tuned model and log to wandb
    """
    print("\nTesting fine-tuned model...")

    # Initialize wandb for testing
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}-test",
        job_type="evaluation"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Test prompts
    test_cases = [
        {
            "name": "EUR Headwind",
            "transcript": """In Q1, we experienced a 4% revenue headwind from currency movements, 
            primarily due to USD strength versus EUR. Our European operations represent approximately 
            35% of total revenue, and we expect this headwind to continue into Q2."""
        },
        {
            "name": "JPY Benefit",
            "transcript": """Our Japan operations benefited from JPY weakness this quarter, with a 3% 
            tailwind to operating margins. We expect volatility to continue but current trends favor 
            our margin profile in the region."""
        },
        {
            "name": "Hedged Exposure",
            "transcript": """We have hedged approximately 75% of our EUR exposure for the next two quarters. 
            While we saw some currency headwinds in Q1, our hedging strategy limits near-term impact."""
        }
    ]
    
    # Generate and log results
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    test_results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        messages = [
            {"role": "user", "content": f"Extract forex trading signals from this earnings call transcript. Return a structured JSON response.\n\nTranscript:\n{test_case['transcript']}"}
        ]
        
        response = pipe(
            messages,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        output = response[0]["generated_text"][-1]["content"]
        print(f"Model Output:\n{output}\n")
        
        # Try to parse as JSON for validation
        try:
            import json
            parsed = json.loads(output)
            is_valid_json = True
            print("Valid JSON output")
        except:
            is_valid_json = False
            print("Invalid JSON output")
        
        test_results.append({
            "test_case": test_case["name"],
            "transcript": test_case["transcript"],
            "model_output": output,
            "is_valid_json": is_valid_json
        })

    # Log test results to wandb
    wandb.log({
        "test_results": wandb.Table(
            columns=["Test Case", "Transcript", "Model Output", "Valid JSON"],
            data=[[r["test_case"], r["transcript"][:100] + "...", r["model_output"], r["is_valid_json"]] for r in test_results]
        )
    })

    # Calculate success rate
    success_rate = sum(1 for r in test_results if r["is_valid_json"]) / len(test_results)
    wandb.log({"json_success_rate": success_rate})
    
    print(f"\n{'='*60}")
    print(f"JSON Success Rate: {success_rate*100:.1f}%")
    print(f"{'='*60}")
    
    wandb.finish()


if __name__ == "__main__":
    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable not set")
        print("You may need to authenticate with Hugging Face")
        print("Run: huggingface-cli login")
        print()

    try:
        wandb.login()
        print("Logged in to Weights & Biases")
    except:
        print("Not logged in to Weights & Biases")
        print("Run: wandb login")
        print()
        
    # Run training
    trainer = train()
    
    # Test the model
    print("\nRunning test evaluation...")
    test_model()
