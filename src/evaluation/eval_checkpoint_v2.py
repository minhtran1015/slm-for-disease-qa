#!/usr/bin/env python3
"""
Evaluate Checkpoint Script for Vietnamese Medical QA
=====================================================
Evaluates the checkpoint-4000 model on test data

Usage:
    modal run eval_checkpoint_v2.py
"""

import modal
from typing import Dict, Any, List

# Modal configuration
app = modal.App("vietnamese-medical-eval-v2")

volume = modal.Volume.from_name("medical-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    gpu="A10G",  # Cheaper GPU for evaluation
    image=image,
    volumes={"/vol": volume},
    timeout=1800,  # 30 minutes
)
def evaluate_checkpoint(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate the checkpoint on test samples
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login
    
    # Login
    hf_token = "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"
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
    
    print(f"🚀 Evaluating {len(test_samples)} samples")
    
    # Paths
    MODEL_NAME = "google/gemma-3-1b-it"
    CHECKPOINT_PATH = "/vol/models/gemma-1b-medical-v2/final"
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization
    print("⚙️  Setting up quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print(f"📦 Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    
    # Load LoRA adapter
    print(f"🔧 Loading LoRA adapter from checkpoint...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    
    # Evaluation
    print("\n🔍 Running evaluation...")
    results = []
    correct = 0
    total = 0
    
    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(test_samples)}...")
        
        try:
            messages = sample["messages"]
            user_content = messages[0]["content"]
            expected_answer = messages[1]["content"].strip()
            
            # Format prompt
            chat = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated = generated.strip()
            
            # Extract answer (Đúng/Sai)
            if "Đúng" in generated:
                predicted = "Đúng"
            elif "Sai" in generated:
                predicted = "Sai"
            else:
                predicted = generated[:20]
            
            # Check correctness
            is_correct = predicted == expected_answer
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": user_content[:100] + "..." if len(user_content) > 100 else user_content,
                "expected": expected_answer,
                "predicted": predicted,
                "correct": is_correct,
            })
            
        except Exception as e:
            print(f"   Error on sample {i}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"   Total Samples: {len(test_samples)}")
    print(f"   Evaluated: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    return {
        "total_samples": len(test_samples),
        "evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results[:20],  # First 20 for inspection
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def load_test_data() -> List[Dict[str, Any]]:
    """Load test data from volume"""
    import json
    
    test_file = "/vol/train_data_v2/test.jsonl"
    
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"📊 Loaded {len(samples)} test samples")
    return samples


@app.local_entrypoint()
def main():
    print("🔍 Evaluating Checkpoint-4000 on Test Data")
    print("=" * 60)
    
    # Load test data
    print("\n📥 Loading test data...")
    test_samples = load_test_data.remote()
    
    print(f"   Loaded {len(test_samples)} samples")
    
    # Evaluate
    print("\n🚀 Starting evaluation...")
    result = evaluate_checkpoint.remote(test_samples)
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Accuracy: {result['accuracy']*100:.2f}%")
    print(f"   ({result['correct']}/{result['evaluated']} correct)")
    
    # Show some examples
    print("\n📋 Sample Results:")
    for i, r in enumerate(result['results'][:5]):
        status = "✅" if r['correct'] else "❌"
        print(f"   {status} Q: {r['question'][:50]}...")
        print(f"      Expected: {r['expected']}, Predicted: {r['predicted']}")
