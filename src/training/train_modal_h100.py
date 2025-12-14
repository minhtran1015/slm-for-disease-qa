#!/usr/bin/env python3
"""
Modal Training Script for Vietnamese Medical QA with Gemma-1B
=============================================================
Optimized for H100 GPU on Modal.com

Usage:
    # Upload data first
    modal volume put medical-data train_data_v2/train.jsonl /data/train_data_v2/train.jsonl
    modal volume put medical-data train_data_v2/val.jsonl /data/train_data_v2/val.jsonl
    
    # Run training
    modal run train_modal_h100.py
"""

import modal
from typing import Dict, Any

# Modal configuration
app = modal.App("vietnamese-medical-training-v2")

volume = modal.Volume.from_name("medical-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "sentencepiece>=0.1.99",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/vol": volume},
    timeout=14400,  # 4 hours max
)
def train_gemma_medical() -> Dict[str, Any]:
    """
    Train Gemma-1B on Vietnamese Medical QA dataset using QLoRA
    Optimized for H100 GPU with larger batch sizes
    """
    import os
    import json
    import torch
    from datetime import datetime
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from huggingface_hub import login
    
    # Hardcoded HF token for simplicity
    hf_token = "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"
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
    
    print("=" * 70)
    print("🚀 MODAL H100 TRAINING - Vietnamese Medical QA v2")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration - Optimized for H100
    MODEL_NAME = "google/gemma-3-1b-it"
    TRAIN_FILE = "/vol/train_data_v2/train.jsonl"
    VAL_FILE = "/vol/train_data_v2/val.jsonl"
    OUTPUT_DIR = "/vol/models/gemma-1b-medical-v2"
    
    # H100 optimized settings (80GB VRAM allows larger batches)
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # Larger batch on H100
    GRAD_ACCUM = 4  # Effective batch = 32
    NUM_EPOCHS = 3
    LEARNING_RATE = 3e-4
    
    # Check data files
    print(f"\n📂 Checking data files...")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")
    
    # Load data
    print(f"📥 Loading training data...")
    train_data = []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    val_data = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    print(f"   Training samples: {len(train_data):,}")
    print(f"   Validation samples: {len(val_data):,}")
    
    # Calculate training steps
    steps_per_epoch = len(train_data) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Total steps: {total_steps:,}")
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4-bit quantization config
    print(f"⚙️  Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    print(f"📦 Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    print(f"🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize function
    def tokenize_data(examples):
        texts = []
        for item in examples:
            messages = item["messages"]
            user_content = messages[0]["content"]
            model_content = messages[1]["content"]
            
            # Apply Gemma chat template
            chat = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": model_content}
            ]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Create datasets
    print(f"\n🔄 Tokenizing datasets...")
    
    # Process in batches
    train_tokenized = tokenize_data(train_data)
    val_tokenized = tokenize_data(val_data)
    
    train_dataset = Dataset.from_dict({
        "input_ids": train_tokenized["input_ids"].tolist(),
        "attention_mask": train_tokenized["attention_mask"].tolist(),
        "labels": train_tokenized["labels"].tolist(),
    })
    
    val_dataset = Dataset.from_dict({
        "input_ids": val_tokenized["input_ids"].tolist(),
        "attention_mask": val_tokenized["attention_mask"].tolist(),
        "labels": val_tokenized["labels"].tolist(),
    })
    
    print(f"   Train dataset: {len(train_dataset):,} samples")
    print(f"   Val dataset: {len(val_dataset):,} samples")
    
    # Training arguments - optimized for H100
    print(f"\n⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    print(f"\n🏋️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 3600
    
    # Save model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Commit volume
    volume.commit()
    
    # Results
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Training time: {training_duration:.2f} hours")
    print(f"📊 Final train loss: {train_result.training_loss:.4f}")
    print(f"💾 Model saved to: {OUTPUT_DIR}")
    
    return {
        "training_time_hours": training_duration,
        "final_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
        "model_path": OUTPUT_DIR,
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def upload_training_data():
    """Helper to verify data is uploaded"""
    import os
    
    train_file = "/vol/train_data_v2/train.jsonl"
    val_file = "/vol/train_data_v2/val.jsonl"
    
    results = {}
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_count = sum(1 for _ in f)
        results["train"] = train_count
    else:
        results["train"] = "NOT FOUND"
    
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_count = sum(1 for _ in f)
        results["val"] = val_count
    else:
        results["val"] = "NOT FOUND"
    
    print(f"📊 Data verification:")
    print(f"   Train: {results['train']}")
    print(f"   Val: {results['val']}")
    
    return results


@app.local_entrypoint()
def main():
    """
    Main entry point for Modal training
    
    Usage:
        1. Upload data:
           modal volume put medical-data train_data_v2/train.jsonl /train_data_v2/train.jsonl
           modal volume put medical-data train_data_v2/val.jsonl /train_data_v2/val.jsonl
        
        2. Verify data:
           modal run train_modal_h100.py::upload_training_data
        
        3. Run training:
           modal run train_modal_h100.py
    """
    print("🚀 Starting Vietnamese Medical QA Training on Modal H100")
    print("=" * 60)
    
    # Verify data first
    print("\n📋 Verifying training data...")
    data_status = upload_training_data.remote()
    
    if data_status["train"] == "NOT FOUND" or data_status["val"] == "NOT FOUND":
        print("\n❌ Training data not found! Please upload first:")
        print("   modal volume put medical-data train_data_v2/train.jsonl /train_data_v2/train.jsonl")
        print("   modal volume put medical-data train_data_v2/val.jsonl /train_data_v2/val.jsonl")
        return
    
    print(f"\n✅ Data verified: {data_status['train']:,} train, {data_status['val']:,} val")
    
    # Start training
    print("\n🏋️ Starting H100 training...")
    result = train_gemma_medical.remote()
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Time: {result['training_time_hours']:.2f} hours")
    print(f"📊 Loss: {result['final_loss']:.4f}")
    print(f"📍 Steps: {result['total_steps']}")
    print(f"💾 Model: {result['model_path']}")
    print("\n📥 Download model:")
    print(f"   modal volume get medical-data {result['model_path']} ./")
