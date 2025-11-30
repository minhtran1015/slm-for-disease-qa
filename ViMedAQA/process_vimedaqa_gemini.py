#!/usr/bin/env python3
"""
ViMedAQA Gemini Processing Pipeline - Batch Processing with Real-time Output

Processes ViMedAQA samples in batches to generate balanced True/False statements
for Vietnamese medical QA training. Each sample generates exactly 1 true and 1 false statement.

Output format matches ICD10-style JSONL for consistent LLM training.

Usage:
    1. Make sure .env file exists with GEMINI_API_KEY
    2. Run: python process_vimedaqa_gemini.py
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# ================= CONFIGURATION =================
SEED = 42
random.seed(SEED)

GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 1
RETRY_DELAY = 0.5
REQUESTS_PER_MINUTE = 1000
BATCH_SIZE = 10  # Process samples in batches
MAX_CONCURRENT = 5  # Max concurrent API calls

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PARQUET = os.path.join(BASE_DIR, "train-00000-of-00001.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "vimedaqa_yesno_train.jsonl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "vimedaqa_checkpoint.json")
STATS_FILE = os.path.join(BASE_DIR, "vimedaqa_stats.json")

MAX_SAMPLES = 10000   # Test with 50 samples for batch processing validation

INSTRUCTION_TEMPLATES = [
    "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau.",
    "Dựa vào kiến thức y khoa, hãy trả lời Đúng hoặc Sai.",
    "Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa.",
    "Xác định tính đúng sai của nhận định y khoa sau.",
    "Với kiến thức về y học, hãy xác nhận câu sau Đúng hay Sai.",
]


def setup_gemini_api() -> genai.GenerativeModel:
    """Configure and return Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables.\n"
            "Make sure .env file exists with: GEMINI_API_KEY=your-api-key"
        )
    
    genai.configure(api_key=api_key)
    
    # Use minimal config - explicit safety settings were blocking medical content
    # Gemini's default safety mechanisms are sufficient
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    
    return model


def create_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """Create prompt for TRUE statement generation."""
    prompt = f"""Hãy chuyển đổi câu hỏi và câu trả lời sau thành một câu khẳng định đơn giản bằng tiếng Việt:

Câu hỏi: {question}
Trả lời: {answer}

Câu khẳng định:"""
    return prompt


def create_false_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """Create prompt for FALSE statement generation - framed as quiz creation."""
    prompt = f"""Bạn đang tạo bài kiểm tra trắc nghiệm y khoa. Tạo một câu khẳng định SAI (làm lựa chọn nhiễu) về cùng chủ đề:

Thông tin đúng:
Câu hỏi: {question}
Trả lời: {answer}

Hãy tạo MỘT câu khẳng định SAI (thay đổi một chi tiết quan trọng để làm nhiễu). Chỉ viết câu khẳng định:"""
    return prompt


def call_gemini_api(
    model: genai.GenerativeModel,
    prompt: str,
    retries: int = MAX_RETRIES
) -> Optional[str]:
    """Call Gemini API with retry logic."""
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(prompt)
            
            # Debug finish reason if available
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                # finish_reason: 1=STOP, 2=RECITATION, 3=SAFETY
                if finish_reason in [2, 3]:
                    # Content blocked by safety or recitation filter
                    return None
            
            # Try to get text
            if hasattr(response, 'text') and response.text and response.text.strip():
                return response.text.strip()
            
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                
        except Exception as e:
            if attempt < retries:
                print(f"   ⚠️ API Error (Attempt {attempt+1}/{retries+1}): {e}")
                time.sleep(RETRY_DELAY)
    
    return None


def reconcile_checkpoint(checkpoint: Dict[str, Any], output_file: str):
    """Reconcile checkpoint processed_indices with actual output file.
    
    Only marks samples as processed if BOTH true and false statements exist.
    """
    if not Path(output_file).exists():
        return checkpoint
        
    print("   🔍 Reconciling checkpoint with output file...")
    
    # Track true and false statements separately
    samples_with_true = set()
    samples_with_false = set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # statement_id format: "vimedaqa_{df_idx}_yes" or "vimedaqa_{df_idx}_no"
                    stmt_id = data.get("statement_id", "")
                    if stmt_id.startswith("vimedaqa_"):
                        parts = stmt_id.split("_")
                        if len(parts) >= 3 and parts[1].isdigit():
                            df_idx = int(parts[1])
                            stmt_type = parts[2]  # 'yes' or 'no'
                            
                            if stmt_type == "yes":
                                samples_with_true.add(df_idx)
                            elif stmt_type == "no":
                                samples_with_false.add(df_idx)
                except:
                    continue
    except Exception as e:
        print(f"   ⚠️ Error reading output file: {e}")
        return checkpoint
    
    # Only samples with BOTH statements are truly processed
    complete_samples = samples_with_true & samples_with_false
    incomplete_samples = (samples_with_true | samples_with_false) - complete_samples
    
    # Report details
    total_statements = len(samples_with_true) + len(samples_with_false)
    print(f"      Total statements in output: {total_statements}")
    print(f"      Complete samples (both statements): {len(complete_samples)}")
    if incomplete_samples:
        print(f"      ⚠️ Incomplete samples (only one statement): {len(incomplete_samples)}")
        
    # Reconstruct processed_indices based on complete samples only
    new_processed_indices = set()
    if "sampled_indices" in checkpoint and checkpoint["sampled_indices"]:
        sampled = checkpoint["sampled_indices"]
        for df_idx in complete_samples:
            if df_idx < len(sampled):
                new_processed_indices.add(sampled[df_idx])
    else:
        new_processed_indices = complete_samples
        
    old_count = len(checkpoint.get("processed_indices", []))
    new_count = len(new_processed_indices)
    
    if old_count != new_count:
        print(f"   ♻️ Reconciled: {old_count} -> {new_count} processed samples")
        checkpoint["processed_indices"] = list(new_processed_indices)
        save_checkpoint(checkpoint)
        
    return checkpoint


def get_complete_samples_from_output(output_file: str, max_samples: Optional[int] = None) -> tuple:
    """Get set of df_idx that have both yes and no statements in the output file.
    Returns (complete_samples_set, total_statements_count)
    """
    complete = set()
    if not Path(output_file).exists():
        return complete, 0
    
    samples_with_true = set()
    samples_with_false = set()
    total_statements = 0
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    stmt_id = data.get("statement_id", "")
                    if stmt_id.startswith("vimedaqa_"):
                        parts = stmt_id.split("_")
                        if len(parts) >= 3 and parts[1].isdigit():
                            df_idx = int(parts[1])
                            # Only count samples within our processing range
                            if max_samples is None or df_idx < max_samples:
                                stmt_type = parts[2]
                                if stmt_type == "yes":
                                    samples_with_true.add(df_idx)
                                elif stmt_type == "no":
                                    samples_with_false.add(df_idx)
                                total_statements += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"   ⚠️ Error reading output file for calibration: {e}")
        return complete, 0
    
    complete = samples_with_true & samples_with_false
    print(f"   🔍 Found {len(complete)} complete samples, {total_statements} total statements in output file")
    print(f"   📊 Breakdown: {len(samples_with_true)} yes statements, {len(samples_with_false)} no statements")
    return complete, total_statements


def process_single_sample(args):
    """Process a single sample - used for batch processing."""
    model, df_idx, question, answer, sample_id = args
    
    results = {
        'df_idx': df_idx,
        'sample_id': sample_id,
        'true_sample': None,
        'false_sample': None,
        'true_success': False,
        'false_success': False
    }
    
    try:
        # TRUE statement
        true_prompt = create_statement_prompt(question, answer)
        true_response = call_gemini_api(model, true_prompt)
        
        if true_response:
            true_statement = true_response.replace("(Đ/S)", "").strip().rstrip('.')
            results['true_sample'] = {
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
            results['true_success'] = True
        
        # Rate limiting
        time.sleep(60 / REQUESTS_PER_MINUTE)
        
        # FALSE statement
        false_prompt = create_false_statement_prompt(question, answer)
        false_response = call_gemini_api(model, false_prompt)
        
        if false_response:
            false_statement = false_response.replace("(Đ/S)", "").strip().rstrip('.')
            # Clean common prefixes
            for prefix in ["Câu phát biểu sai:", "Lựa chọn nhiễu:", "Câu khẳng định sai:", "Câu khẳng định SAI:"]:
                if false_statement.startswith(prefix):
                    false_statement = false_statement[len(prefix):].strip()
            
            if false_statement:
                results['false_sample'] = {
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
                results['false_success'] = True
        
        # Rate limiting
        time.sleep(60 / REQUESTS_PER_MINUTE)
        
    except Exception as e:
        print(f"\n   ❌ Error processing sample {df_idx}: {e}")
    
    return results


def process_batch(model, batch_data):
    """Process a batch of samples using ThreadPoolExecutor."""
    batch_args = []
    for df_idx, row in batch_data:
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        
        if question and answer:
            sample_id = f"vimedaqa_{df_idx}"
            batch_args.append((model, df_idx, question, answer, sample_id))
    
    if not batch_args:
        return []
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(batch_args))) as executor:
        results = list(executor.map(process_single_sample, batch_args))
    
    return results


def load_checkpoint() -> Dict[str, Any]:
    """Load processing checkpoint if exists."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_indices": [],
        "sampled_indices": None,
        "last_processed": -1,
        "total_samples": 0,
        "successful_true": 0,
        "successful_false": 0,
        "failed": 0,
    }


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save processing checkpoint."""
    checkpoint["timestamp"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def save_sample(sample: Dict, filepath: str):
    """Save single sample to JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False)
        f.write('\n')


def main():
    """Main processing pipeline."""
    
    print("=" * 70)
    print("🚀 ViMedAQA Gemini Processing Pipeline - Batch Processing")
    print("   🔧 With Checkpoint Calibration")
    print("=" * 70)
    print(f"   Mode: Batch processing with {BATCH_SIZE} samples per batch, {MAX_CONCURRENT} concurrent workers")
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
    
    # Load checkpoint - handle both old and new format
    checkpoint = load_checkpoint()
    
    # Use new df_indices format if available, fallback to legacy format
    if "processed_df_indices" in checkpoint:
        processed_count = len(checkpoint["processed_df_indices"])
        print(f"   📍 Resuming from checkpoint: {processed_count} samples already processed (using df_indices)")
    else:
        processed_indices = set(checkpoint.get("processed_indices", []))
        processed_count = len(processed_indices)
        if processed_count > 0:
            print(f"   📍 Resuming from legacy checkpoint: {processed_count} samples (will recalibrate)")

    # Skip old reconcile_checkpoint - we'll do full calibration below

    # Calibrate with actual output file
    print("   🔧 Calibrating with output file...")
    complete_df_indices, total_statements = get_complete_samples_from_output(OUTPUT_FILE, MAX_SAMPLES)
    
    # Store df_indices directly (0-9999) instead of original indices
    # This makes the checkpoint much cleaner and easier to understand
    calibrated_processed = complete_df_indices.copy()
    
    # Always update to calibrated state - use df_indices directly
    if calibrated_processed != set(checkpoint.get("processed_df_indices", [])):
        old_count = len(checkpoint.get("processed_df_indices", []))
        print(f"   ♻️ Calibrated processed df_indices: {old_count} -> {len(calibrated_processed)}")
        if old_count > len(calibrated_processed):
            print(f"   📝 Removing {old_count - len(calibrated_processed)} incomplete samples from checkpoint")
        checkpoint["processed_df_indices"] = list(calibrated_processed)
        
        # Keep legacy processed_indices for compatibility but mark as deprecated
        checkpoint["processed_indices"] = []  # Clear the confusing large indices
        checkpoint["_note"] = "processed_df_indices contains 0-9999 range, processed_indices is deprecated"
        
        save_checkpoint(checkpoint)
    
    print(f"   ✅ Calibration complete: {len(complete_df_indices)} samples ready to continue")


    
    # Random sampling
    if MAX_SAMPLES < len(df):
        sampled_indices = checkpoint.get("sampled_indices", None)
        if sampled_indices is None:
            df_sampled = df.sample(n=MAX_SAMPLES, random_state=SEED)
            sampled_indices = df_sampled.index.tolist()
            checkpoint["sampled_indices"] = sampled_indices
            save_checkpoint(checkpoint)
            print(f"   🎯 Randomly sampled {MAX_SAMPLES:,} samples from {total_samples:,} total")
        else:
            df_sampled = df.loc[sampled_indices]
            print(f"   📍 Resumed with {MAX_SAMPLES:,} pre-sampled indices")
        
        df = df_sampled.reset_index(drop=True)
    else:
        df = df.head(MAX_SAMPLES)
        print(f"   🎯 Using {len(df):,} samples")
    
    # Initialize output file - backup if starting fresh
    if len(complete_df_indices) == 0 and Path(OUTPUT_FILE).exists():
        backup_name = f"{OUTPUT_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(OUTPUT_FILE).rename(backup_name)
        print(f"   📦 Backed up existing file to {backup_name}")
    
    # Processing
    print(f"\n🔄 Processing samples in batches of {BATCH_SIZE} with {MAX_CONCURRENT} concurrent workers...")
    print("-" * 70)
    
    stats = {
        "total_processed": len(complete_df_indices),
        "successful_true": 0,
        "successful_false": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
    }
    
    # Use calibrated complete samples (already validated to be < MAX_SAMPLES)
    processed_df_indices = complete_df_indices.copy()
    
    print(f"   📊 Progress validation: {len(processed_df_indices)}/{MAX_SAMPLES} samples completed")
    print(f"   📋 Sample range: {min(processed_df_indices) if processed_df_indices else 'N/A'} to {max(processed_df_indices) if processed_df_indices else 'N/A'}")
    
    try:
        # Process in batches
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with tqdm(total=len(df), desc="Total progress", unit="sample", initial=len(processed_df_indices)) as pbar:
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(df))
                
                # Prepare batch data - only unprocessed samples
                batch_data = []
                
                for df_idx in range(start_idx, end_idx):
                    if df_idx < len(df) and df_idx not in processed_df_indices:
                        # Needs processing
                        row = df.iloc[df_idx]
                        batch_data.append((df_idx, row))
                
                if not batch_data:
                    # Skip batch if all samples already processed
                    continue
                
                # Silent processing to avoid terminal spam
                
                # Process batch (only unprocessed samples)
                if len(batch_data) > 0:
                    batch_results = process_batch(model, batch_data)
                else:
                    batch_results = []
                
                # Save results and update stats
                for result in batch_results:
                    df_idx = result['df_idx']
                    
                    # Check if BOTH are successful - All or Nothing
                    if (result['true_success'] and result['true_sample'] and 
                        result['false_success'] and result['false_sample']):
                        
                        # Save TRUE sample
                        save_sample(result['true_sample'], OUTPUT_FILE)
                        stats["successful_true"] += 1
                        
                        # Save FALSE sample  
                        save_sample(result['false_sample'], OUTPUT_FILE)
                        stats["successful_false"] += 1
                        
                        # Update processed indices (use df_idx directly)
                        processed_df_indices.add(df_idx)
                            
                    else:
                        stats["failed"] += 1
                        print(f"   ❌ Sample {df_idx}: Failed to generate both statements. Skipping.")
                
                # Update progress for newly processed samples only
                newly_processed = sum(1 for result in batch_results 
                                    if result['true_success'] and result['false_success'])
                pbar.update(newly_processed)
                
                # Update checkpoint after each batch
                stats["total_processed"] = len(processed_df_indices)
                checkpoint["processed_df_indices"] = list(processed_df_indices)
                checkpoint["processed_indices"] = []  # Keep empty for compatibility
                checkpoint["last_processed"] = max([r['df_idx'] for r in batch_results]) if batch_results else -1
                checkpoint.update(stats)
                save_checkpoint(checkpoint)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving progress...")
        # Recalibrate processed indices from output file
        current_complete, _ = get_complete_samples_from_output(OUTPUT_FILE, MAX_SAMPLES)
        checkpoint["processed_df_indices"] = list(current_complete)
        checkpoint["processed_indices"] = []  # Clear legacy indices
        checkpoint["total_processed"] = len(current_complete)
        checkpoint.update(stats)
        save_checkpoint(checkpoint)
        print("   ✅ Progress saved. Run again to resume.")
        return
    
    # Final statistics - recount from actual output file
    final_complete, final_total_statements = get_complete_samples_from_output(OUTPUT_FILE, MAX_SAMPLES)
    final_true_count = len(final_complete)  # Each complete sample has one true
    final_false_count = len(final_complete)  # Each complete sample has one false
    
    stats["end_time"] = datetime.now().isoformat()
    
    print("\n" + "=" * 70)
    print("📊 Processing Complete!")
    print("=" * 70)
    print(f"   Total complete samples: {len(final_complete)}")
    print(f"   TRUE statements (✅): {final_true_count}")
    print(f"   FALSE statements (❌): {final_false_count}")
    print(f"   Total statements: {final_true_count + final_false_count}")
    if Path(OUTPUT_FILE).exists():
        print(f"   Output file: {OUTPUT_FILE} ({Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB)")
    
    # Save final stats
    final_stats = {
        "total_complete_samples": len(final_complete),
        "successful_true": final_true_count,
        "successful_false": final_false_count,
        "total_statements": final_true_count + final_false_count,
        "start_time": stats.get("start_time"),
        "end_time": stats["end_time"]
    }
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    print(f"   Stats saved to: {STATS_FILE}")
    
    # Show sample output
    if Path(OUTPUT_FILE).exists():
        print("\n📝 Sample outputs:")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 4:
                    break
                sample = json.loads(line)
                answer_emoji = "✅" if sample['answer'] == "yes" else "❌"
                print(f"\n   [{i+1}] {answer_emoji} {sample['question'][:60]}...")
                print(f"       → {sample['messages'][2]['content']} ({sample['answer_vi']})")


if __name__ == "__main__":
    main()
