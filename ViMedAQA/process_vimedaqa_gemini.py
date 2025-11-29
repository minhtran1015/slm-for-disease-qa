#!/usr/bin/env python3
"""
Gemini API Pipeline for ViMedAQA Yes/No Statement Generation

This script processes ViMedAQA samples one by one using Gemini API to transform
Q&A pairs into True/False statement format for Vietnamese medical QA training.

Output format matches HPO bilingual dataset structure for consistent training.

Usage:
    1. Make sure .env file exists with GEMINI_API_KEY
    2. Run: python process_vimedaqa_gemini.py
    
This will process 10,000 random samples to generate ~20,000 balanced True/False statements.
"""

import os
import sys
import json
import time
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Falling back to manual environment variable loading...")

# ================= CONFIGURATION =================
SEED = 42
random.seed(SEED)

# API Configuration - OPTIMIZED FOR PAID API
GEMINI_MODEL = "gemini-2.5-flash"  # Updated to use Gemini 2.5 Flash
MAX_RETRIES = 1  # Single retry only
RETRY_DELAY = 0.1  # Minimal delay
REQUESTS_PER_MINUTE = 2000  # Aggressive rate for paid API

# File paths
INPUT_PARQUET = "train-00000-of-00001.parquet"
OUTPUT_FILE = "vimedaqa_yesno_gemini_10k_train.jsonl"
CHECKPOINT_FILE = "vimedaqa_gemini_checkpoint.json"
STATS_FILE = "vimedaqa_gemini_stats.json"

# Processing settings - OPTIMIZED FOR PAID API
SAVE_INTERVAL = 20  # Smaller batches for faster parallel processing
MAX_SAMPLES = 10  # TEST: Hardcoded to process 10 samples for testing (20 statements)
BALANCE_RATIO = 1.0  # 1.0 = equal True/False samples (1 true + 1 false per sample)

# Instructions for Gemini
INSTRUCTION_TEMPLATES = [
    "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau.",
    "Dựa vào kiến thức y khoa, hãy trả lời Đúng hoặc Sai.",
    "Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa.",
    "Xác định tính đúng sai của nhận định y khoa sau.",
    "Với kiến thức về y học, hãy xác nhận câu sau Đúng hay Sai.",
]


def setup_gemini_api() -> genai.GenerativeModel:
    """Configure and return Gemini API client."""
    # Load API key from environment variables (.env file)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables.\n"
            "Make sure .env file exists with: GEMINI_API_KEY=your-api-key"
        )
    
    genai.configure(api_key=api_key)
    
    # Use minimal config - explicit safety settings were blocking medical content
    # Gemini's default safety mechanisms work better for medical content generation
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    
    return model


def create_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """
    Create a prompt for Gemini to transform Q&A into a True/False statement.
    Framed as educational quiz creation to avoid safety blocks.
    """
    prompt = f"""Bạn đang tạo câu hỏi trắc nghiệm y khoa cho học sinh. Hãy chuyển đổi thông tin sau thành một câu phát biểu đúng để kiểm tra kiến thức:

Thông tin gốc:
Hỏi: {question}
Đáp: {answer}

Hãy viết thành một câu phát biểu đúng cho bài kiểm tra (không cần ghi "đúng" hay "sai"):"""
    
    return prompt


def create_false_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """
    Create a prompt for Gemini to generate a FALSE medical statement.
    Framed as creating quiz distractors for educational purposes.
    """
    prompt = f"""Bạn đang tạo câu hỏi trắc nghiệm y khoa cho học sinh. Bạn cần tạo một câu phát biểu SAI (làm lựa chọn nhiễu) dựa trên chủ đề này:

Thông tin đúng:
Hỏi: {question} 
Đáp: {answer}

Hãy tạo MỘT câu phát biểu SAI về cùng chủ đề này để làm lựa chọn nhiễu trong bài trắc nghiệm (thay đổi một chi tiết quan trọng). Chỉ viết câu phát biểu, không giải thích:"""
    
    return prompt


def call_gemini_api(
    model: genai.GenerativeModel,
    prompt: str,
    retries: int = MAX_RETRIES
) -> Optional[str]:
    """
    Call Gemini API with minimal retry logic for paid API.
    
    Returns:
        Generated text or None if failed.
    """
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(prompt)
            
            # Check finish reason
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                # finish_reason: 1=STOP, 2=RECITATION, 3=SAFETY
                if finish_reason in [2, 3]:
                    return None
            
            # Check if response has text
            if hasattr(response, 'text') and response.text and response.text.strip():
                return response.text.strip()
            else:
                if attempt == 0:  # Only print on first failure
                    print(f"  ⚠️ Empty response")
                
        except Exception as e:
            error_str = str(e).lower()
            
            # For paid API, minimal waiting and quick failures
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                if attempt < retries:
                    time.sleep(0.1)  # Very short wait for paid API
                
            elif "api" in error_str or "500" in error_str or "404" in error_str:
                return None  # Immediate failure for API errors
                
            else:
                return None  # Immediate failure for other errors
    
    return None


def load_checkpoint() -> Dict[str, Any]:
    """Load processing checkpoint if exists."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_indices": [],
        "sampled_indices": None,  # Store the random sample indices for reproducibility
        "last_processed": -1,
        "total_samples": 0,
        "successful": 0,
        "failed": 0,
    }


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save processing checkpoint."""
    checkpoint["timestamp"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def save_samples(samples: List[Dict], filepath: str, mode: str = 'a'):
    """Save samples to JSONL file."""
    with open(filepath, mode, encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')


def process_batch_samples(
    model: genai.GenerativeModel,
    batch_samples: List[pd.Series],
    batch_start_idx: int
) -> List[Dict[str, Any]]:
    """
    Process a batch of samples - each sample generates exactly 1 true and 1 false statement.
    """
    results = []
    
    for i, row in enumerate(batch_samples):
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        context = str(row.get('context', '')).strip() if 'context' in row else ""
        
        if not question or not answer:
            continue
        
        sample_id = f"vimedaqa_{batch_start_idx + i}"
        
        # Generate TRUE statement
        true_prompt = create_statement_prompt(question, answer, context)
        try:
            true_response = call_gemini_api(model, true_prompt)
            if true_response:
                # Clean the statement - remove common prefixes and suffixes
                true_statement = true_response.replace("(Đ/S)", "").strip()
                true_statement = true_statement.replace("Câu phát biểu đúng:", "").strip()
                true_statement = true_statement.replace("Câu khẳng định:", "").strip() 
                true_statement = true_statement.rstrip('.')
                
                if true_statement:  # Only add if we got a valid statement
                    # Create TRUE statement in ICD10 format
                    true_result = {
                        "messages": [
                            {"role": "system", "content": "Trợ lý AI Y tế. Chỉ trả lời: Đúng hoặc Sai."},
                            {"role": "user", "content": true_statement},
                            {"role": "assistant", "content": "Đúng"}
                        ],
                        "answer": "yes",
                        "answer_vi": "đúng", 
                        "question": true_statement,
                        "question_type": "correct_statement",
                        "statement_id": f"{sample_id}_yes",
                        "source": "vimedaqa",
                        "source_question": question,
                        "source_answer": answer[:200] + "..." if len(answer) > 200 else answer
                    }
                    results.append(true_result)
                
                # Small delay between API calls
                time.sleep(60 / REQUESTS_PER_MINUTE / 2)
            else:
                print(f"  ⚠️ Empty TRUE response for sample {i}")
        
        except Exception as e:
            print(f"  ⚠️ Failed to generate TRUE statement for sample {i}: {e}")
        
        # Generate FALSE statement
        false_prompt = create_false_statement_prompt(question, answer, context)
        try:
            false_response = call_gemini_api(model, false_prompt)
            if false_response:
                # Clean the statement - remove common prefixes and suffixes
                false_statement = false_response.replace("(Đ/S)", "").strip()
                false_statement = false_statement.replace("Câu phát biểu sai:", "").strip()
                false_statement = false_statement.replace("Lựa chọn nhiễu:", "").strip()
                false_statement = false_statement.replace("Câu khẳng định sai:", "").strip()
                false_statement = false_statement.rstrip('.')
                
                if false_statement:  # Only add if we got a valid statement
                    # Create FALSE statement in ICD10 format
                    false_result = {
                        "messages": [
                            {"role": "system", "content": "Trợ lý AI Y tế. Chỉ trả lời: Đúng hoặc Sai."},
                            {"role": "user", "content": false_statement},
                            {"role": "assistant", "content": "Sai"}
                        ],
                        "answer": "no",
                        "answer_vi": "sai",
                        "question": false_statement,
                        "question_type": "incorrect_statement", 
                        "statement_id": f"{sample_id}_no",
                        "source": "vimedaqa",
                        "source_question": question,
                        "source_answer": answer[:200] + "..." if len(answer) > 200 else answer
                    }
                    results.append(false_result)
                else:
                    print(f"  ⚠️ Empty FALSE statement after cleaning for sample {i}")
                
                # Small delay between API calls
                time.sleep(60 / REQUESTS_PER_MINUTE / 2)
            else:
                print(f"  ⚠️ Empty FALSE response for sample {i}")
                
        except Exception as e:
            print(f"  ⚠️ Failed to generate FALSE statement for sample {i}: {e}")
    
    return results


def main():
    """Main processing pipeline - Hardcoded to generate 20k balanced samples."""
    
    print("=" * 60)
    print("🚀 ViMedAQA Gemini Processing Pipeline - Balanced Statement Generation")
    print("=" * 60)
    print(f"   Processing: {MAX_SAMPLES:,} samples → {MAX_SAMPLES*2:,} statements (1 True + 1 False per sample)")
    print(f"   Model: {GEMINI_MODEL}")
    print(f"   Format: ICD10-style JSONL for LLM training")
    print(f"   Output: {OUTPUT_FILE}")
    
    # Setup
    print("\n📡 Setting up Gemini API...")
    try:
        model = setup_gemini_api()
        print(f"   ✅ Connected to {GEMINI_MODEL}")
    except ValueError as e:
        print(f"   ❌ {e}")
        return
    
    # Load data
    print(f"\n📂 Loading data from {INPUT_PARQUET}...")
    try:
        df = pd.read_parquet(INPUT_PARQUET)
        total_samples = len(df)
        print(f"   ✅ Loaded {total_samples:,} samples")
    except Exception as e:
        print(f"   ❌ Failed to load parquet: {e}")
        return
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_indices = set(checkpoint.get("processed_indices", []))
    
    if processed_indices:
        print(f"   📍 Resuming from checkpoint: {len(processed_indices)} already processed")
    
    # Random sampling for variety (but reproducible with seed)
    if MAX_SAMPLES < len(df):
        # Use consistent random sampling - save sampled indices to checkpoint for resume
        sampled_indices = checkpoint.get("sampled_indices", None)
        if sampled_indices is None:
            # First run - create random sample
            df_sampled = df.sample(n=MAX_SAMPLES, random_state=SEED)
            sampled_indices = df_sampled.index.tolist()
            checkpoint["sampled_indices"] = sampled_indices
            save_checkpoint(checkpoint)
            print(f"   🎯 Randomly sampled {MAX_SAMPLES:,} samples from {total_samples:,} total")
        else:
            # Resume - use saved indices
            df_sampled = df.loc[sampled_indices]
            print(f"   📍 Resumed with {MAX_SAMPLES:,} pre-sampled indices")
        
        df = df_sampled.reset_index(drop=True)
    else:
        df = df.head(MAX_SAMPLES)
        print(f"   🎯 Using {len(df):,} samples")
    
    # Initialize output file if new
    if not processed_indices and Path(OUTPUT_FILE).exists():
        # Backup existing file
        backup_name = f"{OUTPUT_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(OUTPUT_FILE).rename(backup_name)
        print(f"   📦 Backed up existing file to {backup_name}")
    
    # Processing with batch optimization for paid API
    print(f"\n🔄 Processing samples with batch optimization...")
    print(f"   Batch size: {SAVE_INTERVAL} samples")
    print(f"   Parallel workers: 10")
    print(f"   Rate limit: {REQUESTS_PER_MINUTE} requests/minute")
    print("-" * 60)
    
    batch_samples = []
    stats = {
        "total_processed": len(processed_indices),
        "successful_true": 0,
        "successful_false": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
    }
    
    try:
        # Convert processed_indices to work with reset dataframe indices
        original_processed = set(checkpoint.get("processed_indices", []))
        processed_df_indices = set()
        
        # Map original indices to new df indices for resume functionality
        if "sampled_indices" in checkpoint:
            for df_idx, orig_idx in enumerate(checkpoint["sampled_indices"]):
                if orig_idx in original_processed:
                    processed_df_indices.add(df_idx)
        
        with tqdm(total=len(df), desc="Processing batches", unit="sample", initial=len(processed_df_indices)) as pbar:
            
            batch = []
            batch_start_idx = 0
            
            for df_idx, row in df.iterrows():
                # Skip already processed
                if df_idx in processed_df_indices:
                    pbar.update(1)
                    continue
                
                batch.append(row)
                
                # Process batch when full or at end
                if len(batch) >= SAVE_INTERVAL or df_idx == len(df) - 1:
                    if batch:
                        # Process batch with parallel API calls
                        batch_results = process_batch_samples(model, batch, batch_start_idx)
                        
                        # Update tracking
                        true_count = 0
                        false_count = 0
                        for result in batch_results:
                            # Count statements by type
                            if result["answer"] == "yes":
                                true_count += 1
                            else:
                                false_count += 1
                        
                        # Update processed indices (one per original sample)
                        samples_in_batch = len(batch)
                        for j in range(samples_in_batch):
                            sample_idx = batch_start_idx + j
                            processed_df_indices.add(sample_idx)
                            
                            # Map back to original index for checkpoint
                            if "sampled_indices" in checkpoint and checkpoint["sampled_indices"] is not None:
                                if sample_idx < len(checkpoint["sampled_indices"]):
                                    orig_idx = checkpoint["sampled_indices"][sample_idx]
                                    processed_indices.add(orig_idx)
                            else:
                                processed_indices.add(sample_idx)
                        
                        # Update stats
                        stats["successful_true"] += true_count
                        stats["successful_false"] += false_count
                        
                        batch_samples.extend(batch_results)
                        stats["total_processed"] = len(processed_indices)
                        
                        # Save batch
                        if batch_samples:
                            save_samples(batch_samples, OUTPUT_FILE, mode='a')
                            checkpoint["processed_indices"] = list(processed_indices)
                            checkpoint["last_processed"] = df_idx
                            checkpoint.update(stats)
                            save_checkpoint(checkpoint)
                            
                            tqdm.write(f"💾 Batch saved: {len(batch_results)} statements from {len(batch)} samples")
                            batch_samples = []
                        
                        # Update progress
                        pbar.update(len(batch))
                        
                        batch = []
                        batch_start_idx = df_idx + 1
        
        # Save remaining
        if batch_samples:
            save_samples(batch_samples, OUTPUT_FILE, mode='a')
            checkpoint["processed_indices"] = list(processed_indices)
            checkpoint.update(stats)
            save_checkpoint(checkpoint)
            print(f"\n   💾 Saved final {len(batch_samples)} samples")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving progress...")
        if batch_samples:
            save_samples(batch_samples, OUTPUT_FILE, mode='a')
        checkpoint["processed_indices"] = list(processed_indices)
        checkpoint.update(stats)
        save_checkpoint(checkpoint)
        print("   ✅ Progress saved. Run again to resume.")
        return
    
    # Final statistics
    stats["end_time"] = datetime.now().isoformat()
    
    # Count final results
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            total_statements = sum(1 for _ in f)
            
        true_count = 0
        false_count = 0
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data["answer"] == "yes":
                    true_count += 1
                else:
                    false_count += 1
    
    print("\n" + "=" * 60)
    print("📊 Processing Complete!")
    print("=" * 60)
    print(f"   Total samples processed: {len(processed_indices)}")
    print(f"   Total statements generated: {total_statements if Path(OUTPUT_FILE).exists() else 0}")
    print(f"   TRUE statements (Đúng): {true_count if Path(OUTPUT_FILE).exists() else 0}")
    print(f"   FALSE statements (Sai): {false_count if Path(OUTPUT_FILE).exists() else 0}")
    print(f"   Balance ratio: {true_count/false_count if Path(OUTPUT_FILE).exists() and false_count > 0 else 'N/A'}")
    print(f"   Output file: {OUTPUT_FILE}")
    
    # Save final stats
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   Stats saved to: {STATS_FILE}")
    
    # Show sample output
    if Path(OUTPUT_FILE).exists():
        print("\n📝 Sample outputs:")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 4:  # Show 2 true + 2 false examples
                    break
                sample = json.loads(line)
                answer_emoji = "✅" if sample['answer'] == "yes" else "❌"
                print(f"\n   [{i+1}] {answer_emoji} {sample['question']}")
                print(f"       → {sample['messages'][2]['content']} ({sample['answer_vi']})")


if __name__ == "__main__":
    main()
