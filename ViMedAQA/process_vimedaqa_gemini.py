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
    load_dotenv()
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

INPUT_PARQUET = "train-00000-of-00001.parquet"
OUTPUT_FILE = "vimedaqa_yesno_train.jsonl"
CHECKPOINT_FILE = "vimedaqa_checkpoint.json"
STATS_FILE = "vimedaqa_stats.json"

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
                time.sleep(RETRY_DELAY)
    
    return None


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
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_indices = set(checkpoint.get("processed_indices", []))
    
    if processed_indices:
        print(f"   📍 Resuming from checkpoint: {len(processed_indices)} already processed")
    
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
    
    # Initialize output file
    if not processed_indices and Path(OUTPUT_FILE).exists():
        backup_name = f"{OUTPUT_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(OUTPUT_FILE).rename(backup_name)
        print(f"   📦 Backed up existing file to {backup_name}")
    
    # Processing
    print(f"\n🔄 Processing samples in batches of {BATCH_SIZE} with {MAX_CONCURRENT} concurrent workers...")
    print("-" * 70)
    
    stats = {
        "total_processed": len(processed_indices),
        "successful_true": 0,
        "successful_false": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
    }
    
    processed_df_indices = set()
    if "sampled_indices" in checkpoint and checkpoint["sampled_indices"]:
        for df_idx, orig_idx in enumerate(checkpoint["sampled_indices"]):
            if orig_idx in processed_indices:
                processed_df_indices.add(df_idx)
    
    try:
        # Process in batches
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with tqdm(total=len(df), desc="Total progress", unit="sample", initial=len(processed_df_indices)) as pbar:
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(df))
                
                # Prepare batch data
                batch_data = []
                for df_idx in range(start_idx, end_idx):
                    if df_idx in processed_df_indices:
                        continue
                    
                    if df_idx < len(df):
                        row = df.iloc[df_idx]
                        batch_data.append((df_idx, row))
                
                if not batch_data:
                    # Skip empty batch
                    pbar.update(end_idx - start_idx)
                    continue
                
                # Silent processing to avoid terminal spam
                
                # Process batch
                batch_results = process_batch(model, batch_data)
                
                # Save results and update stats
                for result in batch_results:
                    df_idx = result['df_idx']
                    
                    # Save TRUE sample
                    if result['true_success'] and result['true_sample']:
                        save_sample(result['true_sample'], OUTPUT_FILE)
                        stats["successful_true"] += 1
                        # TRUE statement saved silently
                    else:
                        stats["failed"] += 1
                        print(f"   ❌ Sample {df_idx}: TRUE statement failed")
                    
                    # Save FALSE sample  
                    if result['false_success'] and result['false_sample']:
                        save_sample(result['false_sample'], OUTPUT_FILE)
                        stats["successful_false"] += 1
                        # FALSE statement saved silently
                    else:
                        stats["failed"] += 1
                        print(f"   ❌ Sample {df_idx}: FALSE statement failed")
                    
                    # Update processed indices
                    processed_df_indices.add(df_idx)
                    if "sampled_indices" in checkpoint and checkpoint["sampled_indices"]:
                        if df_idx < len(checkpoint["sampled_indices"]):
                            orig_idx = checkpoint["sampled_indices"][df_idx]
                            processed_indices.add(orig_idx)
                    else:
                        processed_indices.add(df_idx)
                
                # Update progress
                pbar.update(len(batch_data))
                
                # Update checkpoint after each batch
                stats["total_processed"] = len(processed_indices)
                checkpoint["processed_indices"] = list(processed_indices)
                checkpoint["last_processed"] = max([r['df_idx'] for r in batch_results]) if batch_results else -1
                checkpoint.update(stats)
                save_checkpoint(checkpoint)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving progress...")
        checkpoint["processed_indices"] = list(processed_indices)
        checkpoint.update(stats)
        save_checkpoint(checkpoint)
        print("   ✅ Progress saved. Run again to resume.")
        return
    
    # Final statistics
    stats["end_time"] = datetime.now().isoformat()
    
    print("\n" + "=" * 70)
    print("📊 Processing Complete!")
    print("=" * 70)
    print(f"   Total samples processed: {stats['total_processed']}")
    print(f"   TRUE statements (✅): {stats['successful_true']}")
    print(f"   FALSE statements (❌): {stats['successful_false']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Total statements: {stats['successful_true'] + stats['successful_false']}")
    if Path(OUTPUT_FILE).exists():
        print(f"   Output file: {OUTPUT_FILE} ({Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB)")
    
    # Save final stats
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
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
