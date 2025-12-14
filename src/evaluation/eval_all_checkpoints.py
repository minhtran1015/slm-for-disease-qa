#!/usr/bin/env python3
"""
Evaluate ALL Checkpoints to find best model
============================================
Tests all available checkpoints on test set to detect overfitting

Usage:
    modal run eval_all_checkpoints.py
"""

import modal
from typing import Dict, Any, List

# Modal configuration
app = modal.App("eval-all-checkpoints")

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


# List of checkpoints to evaluate
CHECKPOINTS = [
    "/vol/models/gemma-1b-medical-v2/checkpoint-3000",
    "/vol/models/gemma-1b-medical-v2/checkpoint-4000",
    "/vol/models/gemma-1b-medical-v2/checkpoint-8000",
    "/vol/models/gemma-1b-medical-v2/checkpoint-8652",
    "/vol/models/gemma-1b-medical-v2/final",
]


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/vol": volume},
    timeout=3600,
)
def evaluate_single_checkpoint(checkpoint_path: str, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a single checkpoint"""
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        return {"checkpoint": checkpoint_path, "error": "NOT FOUND", "accuracy": 0}
    
    # Login
    hf_token = "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"
    login(token=hf_token)
    
    checkpoint_name = checkpoint_path.split("/")[-1]
    print(f"\n{'='*50}")
    print(f"🔍 Evaluating: {checkpoint_name}")
    print(f"{'='*50}")
    
    MODEL_NAME = "google/gemma-3-1b-it"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Evaluate
    correct = 0
    total = 0
    
    for i, sample in enumerate(test_samples):
        if i % 200 == 0:
            print(f"   {checkpoint_name}: {i}/{len(test_samples)}...")
        
        try:
            messages = sample["messages"]
            user_content = messages[0]["content"]
            expected_answer = messages[1]["content"].strip()
            
            chat = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            if "Đúng" in generated:
                predicted = "Đúng"
            elif "Sai" in generated:
                predicted = "Sai"
            else:
                predicted = generated[:20]
            
            if predicted == expected_answer:
                correct += 1
            total += 1
            
        except Exception as e:
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"   ✅ {checkpoint_name}: {accuracy*100:.2f}% ({correct}/{total})")
    
    # Clean up GPU memory
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return {
        "checkpoint": checkpoint_name,
        "path": checkpoint_path,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def load_test_data() -> List[Dict[str, Any]]:
    """Load test data"""
    import json
    
    test_file = "/vol/train_data_v2/test.jsonl"
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"📊 Loaded {len(samples)} test samples")
    return samples


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def list_checkpoints() -> List[str]:
    """List available checkpoints"""
    import os
    
    base_dir = "/vol/models/gemma-1b-medical-v2"
    checkpoints = []
    
    if os.path.exists(base_dir):
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                checkpoints.append(item_path)
    
    print(f"📂 Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"   - {cp.split('/')[-1]}")
    
    return checkpoints


@app.local_entrypoint()
def main():
    print("🔍 Evaluating ALL Checkpoints to Find Best Model")
    print("=" * 60)
    
    # Load test data
    print("\n📥 Loading test data...")
    test_samples = load_test_data.remote()
    print(f"   Loaded {len(test_samples)} samples")
    
    # List checkpoints
    print("\n📂 Listing checkpoints...")
    checkpoints = list_checkpoints.remote()
    
    # Evaluate each checkpoint
    print("\n🚀 Starting evaluation of all checkpoints...")
    results = []
    
    for checkpoint_path in checkpoints:
        result = evaluate_single_checkpoint.remote(checkpoint_path, test_samples)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY - ALL CHECKPOINTS")
    print("=" * 60)
    print(f"{'Checkpoint':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 60)
    
    best_result = None
    for r in sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True):
        if 'error' in r:
            print(f"{r['checkpoint']:<20} {'ERROR':>10}")
        else:
            acc_str = f"{r['accuracy']*100:.2f}%"
            print(f"{r['checkpoint']:<20} {acc_str:>10} {r['correct']:>10} {r['total']:>8}")
            if best_result is None:
                best_result = r
    
    print("-" * 60)
    if best_result:
        print(f"\n🏆 BEST CHECKPOINT: {best_result['checkpoint']}")
        print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")
        print(f"   Path: {best_result['path']}")
