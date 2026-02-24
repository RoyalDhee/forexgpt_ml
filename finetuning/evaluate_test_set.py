#!/usr/bin/env python3
"""
Evaluate ForexGPT on the HELD-OUT TEST SET
Run this ONLY AFTER training is complete
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import wandb

# Configuration
MODEL_PATH = "models/forexgpt-mistral-7b-lora"  # Path to your trained model
TEST_DATA_PATH = "data/labeled/test.json"
WANDB_PROJECT = "forexgpt-finetuning"

def load_test_data(test_path):
    """Load test set"""
    print(f"Loading test data from {test_path}...")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"Found {len(test_data)} test examples")
    return test_data


def extract_ground_truth(example):
    """Extract ground truth from test example"""
    # The assistant message contains the ground truth JSON
    for message in example["messages"]:
        if message["role"] == "assistant":
            return json.loads(message["content"])
    return None


def extract_transcript(example):
    """Extract transcript from test example"""
    for message in example["messages"]:
        if message["role"] == "user":
            # Extract just the transcript part
            content = message["content"]
            if "Transcript:" in content:
                return content.split("Transcript:")[-1].strip()
    return None


def evaluate_model(model_path, test_data):
    """
    Evaluate fine-tuned model on test set
    """
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Metrics
    total_examples = len(test_data)
    correct_direction = 0
    correct_pair = 0
    valid_json = 0
    confidence_errors = []
    
    results = []
    
    print(f"\nEvaluating on {total_examples} test examples...")
    print("="*60)
    
    for idx, example in enumerate(tqdm(test_data, desc="Testing")):
        # Get ground truth
        ground_truth = extract_ground_truth(example)
        transcript = extract_transcript(example)
        
        # Generate prediction
        messages = [
            {"role": "user", "content": f"Extract forex trading signals from this earnings call transcript. Return a structured JSON response.\n\nTranscript:\n{transcript}"}
        ]
        
        try:
            response = pipe(
                messages,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for more deterministic outputs
                do_sample=False,   # Greedy decoding for consistency
            )
            
            prediction_text = response[0]["generated_text"][-1]["content"]
            
            # Try to parse as JSON
            try:
                prediction = json.loads(prediction_text)
                valid_json += 1
                
                # Check direction accuracy
                if prediction.get("direction") == ground_truth.get("direction"):
                    correct_direction += 1
                
                # Check currency pair accuracy
                if prediction.get("currency_pair") == ground_truth.get("currency_pair"):
                    correct_pair += 1
                
                # Track confidence error
                pred_conf = prediction.get("confidence", 0)
                true_conf = ground_truth.get("confidence", 0)
                confidence_errors.append(abs(pred_conf - true_conf))
                
            except json.JSONDecodeError:
                prediction = {"error": "Invalid JSON", "raw": prediction_text}
        
        except Exception as e:
            prediction = {"error": str(e)}
        
        # Store result
        results.append({
            "example_id": idx,
            "transcript": transcript[:100] + "..." if len(transcript) > 100 else transcript,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct_direction": prediction.get("direction") == ground_truth.get("direction"),
            "correct_pair": prediction.get("currency_pair") == ground_truth.get("currency_pair"),
        })
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
    direction_accuracy = (correct_direction / total_examples) * 100
    pair_accuracy = (correct_pair / total_examples) * 100
    json_validity = (valid_json / total_examples) * 100
    avg_confidence_error = sum(confidence_errors) / len(confidence_errors) if confidence_errors else 0
    
    metrics = {
        "test_total_examples": total_examples,
        "test_direction_accuracy": direction_accuracy,
        "test_pair_accuracy": pair_accuracy,
        "test_json_validity": json_validity,
        "test_avg_confidence_error": avg_confidence_error,
        "test_correct_direction": correct_direction,
        "test_correct_pair": correct_pair,
        "test_valid_json": valid_json,
    }
    
    print(f"Total Examples:           {total_examples}")
    print(f"Valid JSON Responses:     {valid_json}/{total_examples} ({json_validity:.1f}%)")
    print(f"Correct Direction:        {correct_direction}/{total_examples} ({direction_accuracy:.1f}%)")
    print(f"Correct Currency Pair:    {correct_pair}/{total_examples} ({pair_accuracy:.1f}%)")
    print(f"Avg Confidence Error:     {avg_confidence_error:.3f}")
    print("="*60)
    
    # Pass/Fail criteria
    print("\nMeeting Success Criteria:")
    print(f"  Direction Accuracy > 75%:  {'✓ PASS' if direction_accuracy > 75 else '✗ FAIL'} ({direction_accuracy:.1f}%)")
    print(f"  JSON Validity > 90%:       {'✓ PASS' if json_validity > 90 else '✗ FAIL'} ({json_validity:.1f}%)")
    print("="*60)
    
    return metrics, results


def log_to_wandb(metrics, results):
    """Log test results to Weights & Biases"""
    print("\nLogging results to Weights & Biases...")
    
    wandb.init(
        project=WANDB_PROJECT,
        name="final-test-evaluation",
        job_type="evaluation"
    )
    
    # Log summary metrics
    wandb.log(metrics)
    
    # Create detailed results table
    table_data = []
    for r in results[:20]:  # First 20 for readability
        table_data.append([
            r["example_id"],
            r["transcript"],
            r["ground_truth"].get("direction", "N/A"),
            r["prediction"].get("direction", "N/A"),
            "✓" if r["correct_direction"] else "✗",
            r["ground_truth"].get("currency_pair", "N/A"),
            r["prediction"].get("currency_pair", "N/A"),
            "✓" if r["correct_pair"] else "✗",
        ])
    
    wandb.log({
        "test_results_sample": wandb.Table(
            columns=["ID", "Transcript", "True Direction", "Pred Direction", "Correct?", 
                    "True Pair", "Pred Pair", "Correct?"],
            data=table_data
        )
    })
    
    # Save full results as artifact
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    artifact = wandb.Artifact("test-results", type="evaluation")
    artifact.add_file("test_results.json")
    wandb.log_artifact(artifact)
    
    print(f"✓ Results logged to: {wandb.run.get_url()}")
    wandb.finish()


if __name__ == "__main__":
    print("="*60)
    print("FOREXGPT TEST SET EVALUATION")
    print("="*60)
    print("\n⚠️  IMPORTANT: Run this ONLY AFTER training is complete!")
    print("This evaluates on the held-out test set that was never seen during training.\n")
    
    # Load test data
    test_data = load_test_data(TEST_DATA_PATH)
    
    # Run evaluation
    metrics, results = evaluate_model(MODEL_PATH, test_data)
    
    # Log to wandb
    log_to_wandb(metrics, results)
    
    print("\n✓ Test evaluation complete!")
    print(f"✓ Results saved to test_results.json")
    print(f"✓ Use these metrics in your defense presentation")
