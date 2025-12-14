#!/usr/bin/env python3
"""
FULLY MONITORED Training script for Vietnamese Medical Dataset with Gemma-1B
Complete error logging, progress tracking, and crash handling
Only works on Window machine
"""

import json
import torch
import os
import time
import traceback
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Initialize session ID for all log files
SESSION_ID = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Log file paths
MAIN_LOG = f"logs/{SESSION_ID}_main.log"
ERROR_LOG = f"logs/{SESSION_ID}_errors.log"
PROGRESS_LOG = f"logs/{SESSION_ID}_progress.log"
SYSTEM_LOG = f"logs/{SESSION_ID}_system.log"

def log_to_file(filepath, message, include_timestamp=True):
    """Helper function to log to specific file"""
    timestamp = f"{datetime.now().isoformat()} - " if include_timestamp else ""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp}{message}\n")

def log_info(message):
    """Log general information"""
    print(f"ℹ️ {message}")
    log_to_file(MAIN_LOG, f"INFO - {message}")

def log_error(message, exception=None):
    """Log error with full details"""
    print(f"❌ ERROR: {message}")
    log_to_file(MAIN_LOG, f"ERROR - {message}")
    
    if exception:
        error_details = f"ERROR - {message}\nException: {type(exception).__name__}: {str(exception)}\nTraceback:\n{traceback.format_exc()}\n{'='*80}"
    else:
        error_details = f"ERROR - {message}\n{'='*80}"
    
    log_to_file(ERROR_LOG, error_details)

def log_progress(step, total_steps, loss, eta_hours, gpu_memory=None):
    """Log training progress"""
    progress_pct = (step / total_steps) * 100
    progress_msg = f"Step {step}/{total_steps} ({progress_pct:.1f}%) | Loss: {loss:.4f} | ETA: {eta_hours:.1f}h"
    
    if gpu_memory:
        progress_msg += f" | GPU: {gpu_memory:.1f}GB"
    
    print(f"📊 {progress_msg}")
    log_to_file(PROGRESS_LOG, progress_msg)

def log_system(message):
    """Log system/hardware information"""
    print(f"🖥️ {message}")
    log_to_file(SYSTEM_LOG, message)

def save_error_summary(error_message, full_traceback):
    """Save comprehensive error summary"""
    error_summary = {
        'session_id': SESSION_ID,
        'timestamp': datetime.now().isoformat(),
        'error_message': error_message,
        'traceback': full_traceback,
        'log_files': {
            'main': MAIN_LOG,
            'errors': ERROR_LOG,
            'progress': PROGRESS_LOG,
            'system': SYSTEM_LOG
        }
    }
    
    error_summary_file = f"logs/{SESSION_ID}_error_summary.json"
    with open(error_summary_file, 'w', encoding='utf-8') as f:
        json.dump(error_summary, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Error summary saved: {error_summary_file}")

# Configuration - OPTIMIZED FOR 16GB VRAM
MODEL_NAME = "google/gemma-3-1b-it"
DATASET_PATH = "train_data/train.jsonl"
OUTPUT_DIR = "./results/gemma-1b-vietnamese-medical-16gb"
MAX_LENGTH = 512  # Reduced from 1024 to save memory
BATCH_SIZE = 4    # Reduced from 16 to save memory
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4

# Hardware optimizations - MEMORY FOCUSED FOR 16GB VRAM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Memory optimization for 16GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

class MonitoredTrainer(Trainer):
    """Enhanced trainer with monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_times = []
        self.losses = []
        self.start_time = time.time()
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        step_start = time.time()
        try:
            # Handle the new transformers API that passes num_items_in_batch
            if num_items_in_batch is not None:
                result = super().training_step(model, inputs, num_items_in_batch)
            else:
                result = super().training_step(model, inputs)
                
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            
            # Fix: result is a scalar tensor, not a dict
            if isinstance(result, dict):
                loss_value = float(result['loss'])
            else:
                loss_value = float(result)  # result is already the loss tensor
            
            self.losses.append(loss_value)
            
            # Progress logging every 100 steps
            if self.state.global_step % 100 == 0:
                self.log_progress()
                
            # GPU memory every 100 steps
            if self.state.global_step % 100 == 0:
                self.log_gpu_memory()
                
            return result
            
        except Exception as e:
            error_msg = f"Training step {self.state.global_step} failed: {str(e)}"
            log_error(error_msg, e)
            raise
            
    def log_progress(self):
        """Log detailed progress"""
        if not self.step_times:
            return
            
        avg_time = np.mean(self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times)
        avg_loss = np.mean(self.losses[-100:] if len(self.losses) >= 100 else self.losses) if self.losses else 0
        remaining_steps = self.state.max_steps - self.state.global_step
        eta_hours = (remaining_steps * avg_time) / 3600
        
        log_progress(
            step=self.state.global_step,
            total_steps=self.state.max_steps,
            loss=avg_loss,
            eta_hours=eta_hours
        )
        
    def log_gpu_memory(self):
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            gpu_info = f"GPU Memory - Allocated: {memory_allocated:.1f}GB | Reserved: {memory_reserved:.1f}GB | Total: {memory_total:.1f}GB"
            log_system(gpu_info)
            
    def log(self, logs, start_time=None):
        """Override log method to handle new transformers API"""
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Only log metrics every 100 steps to reduce spam
        if self.state.global_step % 100 == 0:
            if 'loss' in logs:
                log_info(f"Step {self.state.global_step} metrics - Loss: {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                log_info(f"Step {self.state.global_step} evaluation - Loss: {logs['eval_loss']:.4f}")

def main():
    start_time = time.time()
    
    # Initialize logging
    log_info("="*80)
    log_info(f"STARTING MONITORED VIETNAMESE MEDICAL QA TRAINING - SESSION: {SESSION_ID}")
    log_info("="*80)
    
    print(f"📝 Session ID: {SESSION_ID}")
    print(f"📁 Log files will be saved in logs/ folder")
    print(f"   📄 Main log: {MAIN_LOG}")
    print(f"   🚨 Error log: {ERROR_LOG}")
    print(f"   📊 Progress log: {PROGRESS_LOG}")
    print(f"   🖥️ System log: {SYSTEM_LOG}")
    
    try:
        # Environment validation
        log_info("Validating training environment...")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! GPU required for training.")
            
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
            
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Hardware info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🔥 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        log_system(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        print(f"🚀 Starting training...")
        print(f"📊 Dataset: {DATASET_PATH}")
        print(f"🤖 Model: {MODEL_NAME}")
        print(f"💾 Output: {OUTPUT_DIR}")
        
        log_info(f"Training config - Dataset: {DATASET_PATH}, Model: {MODEL_NAME}, Batch: {BATCH_SIZE}, Max length: {MAX_LENGTH}")
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        log_info("Loading tokenizer")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token="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",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log_info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Model loading
        print("🧠 Loading model with 4-bit quantization...")
        log_info("Loading model with quantization")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            token="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",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
        log_info("Model prepared for k-bit training")
        
        # LoRA setup
        print("🔧 Setting up LoRA...")
        log_info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Reduced from 64 to save memory
            lora_alpha=64,  # Reduced from 128 to save memory
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        trainable_pct = 100 * trainable_params / all_params
        
        print(f"🔧 LoRA setup complete:")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   All params: {all_params:,}")
        print(f"   Trainable %: {trainable_pct:.4f}")
        
        log_info(f"LoRA setup - Trainable: {trainable_params:,} ({trainable_pct:.4f}%), Total: {all_params:,}")
        
        # Dataset loading
        print("📚 Loading dataset...")
        log_info(f"Loading dataset from {DATASET_PATH}")
        
        data = []
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    # Handle Gemma chat format with messages
                    if "messages" in item and len(item["messages"]) >= 2:
                        user_message = item["messages"][0]["content"]
                        model_response = item["messages"][1]["content"]
                        
                        # Apply chat template
                        chat = [
                            {"role": "user", "content": user_message},
                            {"role": "model", "content": model_response}
                        ]
                        text = tokenizer.apply_chat_template(chat, tokenize=False)
                        data.append({"text": text})
                    
                    # Fallback for old format
                    elif "instruction" in item and "input" in item and "output" in item:
                        question = f"{item['instruction']} {item['input']}"
                        answer = item["output"]
                        text = f"Question: {question}\nAnswer: {answer}"
                        data.append({"text": text})
                        
                    if line_num % 10000 == 0:
                        print(f"   Loaded {line_num:,} samples...")
                        log_info(f"Dataset loading progress: {line_num:,} samples")
                        
                except json.JSONDecodeError as e:
                    log_error(f"Skipping invalid JSON at line {line_num}: {str(e)}")
                    continue
        
        print(f"✅ Loaded {len(data):,} samples")
        log_info(f"Successfully loaded {len(data):,} samples")
        
        # Tokenization
        print("🔤 Tokenizing dataset...")
        log_info("Starting tokenization")
        
        dataset = Dataset.from_list(data)
        
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_overflowing_tokens=False,
            )
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=1,  # Fixed: Use single process to avoid Windows multiprocessing issues
            remove_columns=dataset.column_names,
        )
        
        # Train/eval split - Use dedicated validation file
        print("📊 Loading validation dataset...")
        
        # Load validation data
        val_data = []
        val_file = "train_data/val.jsonl"
        try:
            with open(val_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # Handle Gemma chat format with messages
                        if "messages" in item and len(item["messages"]) >= 2:
                            user_message = item["messages"][0]["content"]
                            model_response = item["messages"][1]["content"]
                            
                            # Apply chat template
                            chat = [
                                {"role": "user", "content": user_message},
                                {"role": "model", "content": model_response}
                            ]
                            text = tokenizer.apply_chat_template(chat, tokenize=False)
                            val_data.append({"text": text})
                    except json.JSONDecodeError as e:
                        log_error(f"Skipping invalid JSON in val.jsonl at line {line_num}: {str(e)}")
                        continue
            
            print(f"✅ Loaded {len(val_data)} validation samples from {val_file}")
            log_info(f"Successfully loaded {len(val_data)} validation samples")
            
            # Create datasets
            train_dataset = tokenized_dataset
            val_dataset = Dataset.from_list(val_data)
            
            # Tokenize validation dataset
            eval_dataset = val_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=val_dataset.column_names,
            )
            
        except FileNotFoundError:
            log_error(f"Validation file {val_file} not found! Using train/test split instead")
            print(f"⚠️  {val_file} not found! Using 10% train/test split instead")
            split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        print(f"📊 Training samples: {len(train_dataset):,}")
        print(f"📊 Evaluation samples: {len(eval_dataset):,}")
        log_info(f"Dataset split - Training: {len(train_dataset):,}, Evaluation: {len(eval_dataset):,}")
        
        # Training setup
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,  # Increased to maintain effective batch size
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=2000,  # Less frequent to save memory
            save_steps=2000,
            save_total_limit=3,  # Fewer checkpoints to save disk space
            load_best_model_at_end=False,
            warmup_steps=100,  # Reduced warmup
            lr_scheduler_type="cosine",
            optim="adamw_torch",  # Standard optimizer (fused uses more memory)
            dataloader_pin_memory=False,  # Disable to save memory
            dataloader_num_workers=2,  # Reduced workers for memory
            report_to=[],
            disable_tqdm=False,
            bf16=True,
            gradient_checkpointing=True,  # Enable to save memory
            remove_unused_columns=False,
            max_grad_norm=1.0,  # Add gradient clipping
        )
        
        # Create trainer
        print("🏋️ Setting up trainer...")
        trainer = MonitoredTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        total_steps = len(train_dataset) // (BATCH_SIZE * 2) * NUM_EPOCHS  # Approximate
        print(f"🚀 Starting training...")
        print(f"📊 Approximate total steps: {total_steps:,}")
        log_info(f"Training started - Estimated {total_steps:,} steps")
        
        trainer.train()
        
        # Save model
        print("💾 Saving final model...")
        log_info("Saving final model")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Final evaluation
        print("📈 Running final evaluation...")
        eval_results = trainer.evaluate()
        final_loss = eval_results.get('eval_loss', 'N/A')
        
        print(f"✅ Training completed successfully!")
        print(f"📊 Final evaluation loss: {final_loss}")
        
        total_time = time.time() - start_time
        log_info(f"Training completed successfully in {total_time/3600:.2f} hours")
        log_info(f"Final evaluation loss: {final_loss}")
        
    except Exception as e:
        error_message = f"Training failed: {str(e)}"
        full_traceback = traceback.format_exc()
        
        log_error(error_message, e)
        save_error_summary(error_message, full_traceback)
        
        print(f"❌ {error_message}")
        print(f"📝 Detailed error logs saved to: {ERROR_LOG}")
        print(f"📋 Error summary saved to: logs/{SESSION_ID}_error_summary.json")
        
        raise
        
    finally:
        total_time = time.time() - start_time
        log_info(f"Training session ended after {total_time/3600:.2f} hours")
        log_info("="*80)
        
        print(f"📁 All logs saved in logs/ folder:")
        print(f"   📄 {MAIN_LOG}")
        print(f"   🚨 {ERROR_LOG}")  
        print(f"   📊 {PROGRESS_LOG}")
        print(f"   🖥️ {SYSTEM_LOG}")

if __name__ == "__main__":
    main()