#!/usr/bin/env python3
"""
Resume Training Script for Vietnamese Medical QA with Gemma-1B
==============================================================
Resumes training from checkpoint-4000

Usage:
    # Upload checkpoint first
    modal volume put medical-data checkpoint_v2/checkpoint-4000 /models/gemma-1b-medical-v2/checkpoint-4000
    
    # Run resume training
    modal run train_modal_resume.py
"""

import modal
from typing import Dict, Any

# Modal configuration
app = modal.App("vietnamese-medical-training-resume")

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
def resume_training() -> Dict[str, Any]:
    """
    Resume training from checkpoint-4000
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
    from peft import PeftModel, prepare_model_for_kbit_training
    from huggingface_hub import login
    
    # Hardcoded HF token
    hf_token = "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"
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
    
    print("=" * 70)
    print("🔄 RESUMING TRAINING - Vietnamese Medical QA v2")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    MODEL_NAME = "google/gemma-3-1b-it"
    CHECKPOINT_PATH = "/vol/models/gemma-1b-medical-v2/checkpoint-4000"
    TRAIN_FILE = "/vol/train_data_v2/train.jsonl"
    VAL_FILE = "/vol/train_data_v2/val.jsonl"
    OUTPUT_DIR = "/vol/models/gemma-1b-medical-v2"
    
    # Training settings
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 3e-4
    
    # Check files
    print(f"\n📂 Checking files...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    
    print(f"   ✅ Checkpoint: {CHECKPOINT_PATH}")
    
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
    
    # Load tokenizer from checkpoint
    print(f"\n📦 Loading tokenizer from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
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
    
    # Load base model
    print(f"📦 Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )
    
    # Prepare for k-bit training BEFORE loading adapter
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Load LoRA adapter from checkpoint
    print(f"🔧 Loading LoRA adapter from checkpoint...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH, is_trainable=True)
    
    # Ensure model is in training mode and gradients are enabled
    model.train()
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # Tokenize function
    def tokenize_data(examples):
        texts = []
        for item in examples:
            messages = item["messages"]
            user_content = messages[0]["content"]
            model_content = messages[1]["content"]
            
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
    
    # Training arguments
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
        resume_from_checkpoint=CHECKPOINT_PATH,  # Resume from checkpoint!
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    print(f"\n🏋️ Resuming training from checkpoint-4000...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Resume training
    start_time = datetime.now()
    train_result = trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 3600
    
    # Save final model
    print(f"\n💾 Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    # Commit volume
    volume.commit()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Training time: {training_duration:.2f} hours")
    print(f"📊 Final train loss: {train_result.training_loss:.4f}")
    print(f"💾 Model saved to: {OUTPUT_DIR}/final")
    
    return {
        "training_time_hours": training_duration,
        "final_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
        "model_path": f"{OUTPUT_DIR}/final",
    }


@app.local_entrypoint()
def main():
    print("🔄 Resuming Vietnamese Medical QA Training from checkpoint-4000")
    print("=" * 60)
    
    result = resume_training.remote()
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Time: {result['training_time_hours']:.2f} hours")
    print(f"📊 Loss: {result['final_loss']:.4f}")
    print(f"📍 Steps: {result['total_steps']}")
    print(f"💾 Model: {result['model_path']}")
    print("\n📥 Download model:")
    print(f"   modal volume get medical-data {result['model_path']} ./")
