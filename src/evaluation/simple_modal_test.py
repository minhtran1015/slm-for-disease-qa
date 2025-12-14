#!/usr/bin/env python3
"""
Simple Modal test script to evaluate the model on test.jsonl
This bypasses the complex vLLM setup and uses basic transformers inference
"""

import modal
import json
from typing import List, Dict, Any

# Create Modal app
app = modal.App("simple-medical-inference")

# Use a basic image with transformers
image = modal.Image.from_registry("python:3.11").pip_install(
    "transformers", "torch", "peft", "accelerate", "bitsandbytes"
)

# Create volume reference
volume = modal.Volume.from_name("slm_disease_qa-volume", create_if_missing=False)

@app.function(
    gpu="A10G",  # Use A10G instead of H100 for faster startup
    image=image,
    volumes={"/vol": volume},
    timeout=1200  # 20 minutes
)
def evaluate_model(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple model evaluation using transformers
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import os
    
    print(f"🚀 Starting evaluation of {len(test_samples)} samples")
    
    # Model paths
    model_path = "/vol/models/gemma-1b-finetuned"
    base_model_path = "google/gemma-1b-it"
    
    print(f"📁 Loading model from {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Evaluate samples
    correct = 0
    total = 0
    results = []
    
    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"📊 Processing sample {i+1}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%)")
        
        try:
            # Get user content and expected answer
            user_content = sample["messages"][0]["content"]
            expected = sample["messages"][1]["content"]
            
            # Format input
            messages = [{"role": "user", "content": user_content}]
            formatted_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate response
            inputs = tokenizer(
                formatted_input, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=3,  # Only need "Đúng" or "Sai"
                    do_sample=False, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # Classify response
            if response.lower().startswith("đúng"):
                prediction = "Đúng"
            elif response.lower().startswith("sai"):
                prediction = "Sai"
            else:
                prediction = "Unknown"
            
            # Check correctness
            is_correct = prediction == expected
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "sample_id": i,
                "question": user_content,
                "expected": expected,
                "predicted": prediction,
                "raw_response": response,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            results.append({
                "sample_id": i,
                "error": str(e)
            })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    answer_rate = total / len(test_samples)
    
    print(f"\n🎯 Evaluation Results:")
    print(f"   Total Samples: {len(test_samples)}")
    print(f"   Answered: {total} ({answer_rate*100:.1f}%)")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    return {
        "total_samples": len(test_samples),
        "answered_samples": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "answer_rate": answer_rate,
        "results": results[:10],  # Only return first 10 detailed results to avoid size limits
        "summary": f"Accuracy: {accuracy*100:.1f}% ({correct}/{total} answered, {answer_rate*100:.1f}% answer rate)"
    }

@app.local_entrypoint()
def main(test_file: str = "./train_data/test.jsonl"):
    """
    Load test data and run evaluation
    """
    print(f"📊 Loading test data from {test_file}")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # Run evaluation
    result = evaluate_model.remote(test_data)
    
    # Print results
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION RESULTS")
    print("="*60)
    print(result["summary"])
    print(f"📈 Performance Level: ", end="")
    
    accuracy = result["accuracy"] * 100
    if accuracy >= 75:
        print("🏆 EXCELLENT")
    elif accuracy >= 65:
        print("🥇 VERY GOOD")
    elif accuracy >= 55:
        print("🥈 GOOD")
    else:
        print("⚠️ NEEDS IMPROVEMENT")
    
    print("="*60)
    
    return result

if __name__ == "__main__":
    result = main()