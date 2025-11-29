#!/usr/bin/env python3
"""
ViMedAQA Gemini Processing Pipeline - One-by-one Processing with Real-time Output

Processes ViMedAQA samples sequentially to generate balanced True/False statements
for Vietnamese medical QA training. Each sample generates exactly 1 true and 1 false statement.

Output format matches ICD10-style JSONL for consistent LLM training.

Usage:
    1. Make sure .env file exists with GEMINI_API_KEY
    2. Run: python process_vimedaqa_gemini_v2.py
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
REQUESTS_PER_MINUTE = 2000

INPUT_PARQUET = "train-00000-of-00001.parquet"
OUTPUT_FILE = "vimedaqa_yesno_gemini_10k_train.jsonl"
CHECKPOINT_FILE = "vimedaqa_gemini_checkpoint.json"
STATS_FILE = "vimedaqa_gemini_stats.json"

MAX_SAMPLES = 10  # TEST: Process 10 samples (20 statements total)

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
    print("🚀 ViMedAQA Gemini Processing Pipeline - One-by-One Processing")
    print("=" * 70)
    print(f"   Mode: Sequential processing with real-time output")
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
    print(f"\n🔄 Processing samples one by one with real-time output...")
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
        with tqdm(total=len(df), desc="Total progress", unit="sample", initial=len(processed_df_indices)) as pbar:
            
            for df_idx, row in df.iterrows():
                # Skip already processed
                if df_idx in processed_df_indices:
                    pbar.update(1)
                    continue
                
                question = str(row.get('question', '')).strip()
                answer = str(row.get('answer', '')).strip()
                
                if not question or not answer:
                    print(f"\n   ⏭️  Sample {df_idx}: Skipped (missing data)")
                    pbar.update(1)
                    continue
                
                sample_id = f"vimedaqa_{df_idx}"
                print(f"\n   📝 Sample {df_idx}: {question[:50]}..." if len(question) > 50 else f"\n   📝 Sample {df_idx}: {question}")
                
                # TRUE statement
                print(f"      🟢 TRUE:  ", end="", flush=True)
                true_prompt = create_statement_prompt(question, answer)
                true_response = call_gemini_api(model, true_prompt)
                
                if true_response:
                    true_statement = true_response.replace("(Đ/S)", "").strip().rstrip('.')
                    true_sample = {
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
                    save_sample(true_sample, OUTPUT_FILE)
                    stats["successful_true"] += 1
                    print("✅")
                    time.sleep(60 / REQUESTS_PER_MINUTE)
                else:
                    print("❌ (no response)")
                    stats["failed"] += 1
                
                # FALSE statement
                print(f"      🔴 FALSE: ", end="", flush=True)
                false_prompt = create_false_statement_prompt(question, answer)
                false_response = call_gemini_api(model, false_prompt)
                
                if false_response:
                    false_statement = false_response.replace("(Đ/S)", "").strip().rstrip('.')
                    # Clean common prefixes
                    for prefix in ["Câu phát biểu sai:", "Lựa chọn nhiễu:", "Câu khẳng định sai:", "Câu khẳng định SAI:"]:
                        if false_statement.startswith(prefix):
                            false_statement = false_statement[len(prefix):].strip()
                    
                    if false_statement:
                        false_sample = {
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
                        save_sample(false_sample, OUTPUT_FILE)
                        stats["successful_false"] += 1
                        print("✅")
                    else:
                        print("❌ (empty after cleaning)")
                        stats["failed"] += 1
                    time.sleep(60 / REQUESTS_PER_MINUTE)
                else:
                    print("❌ (no response)")
                    stats["failed"] += 1
                
                # Update checkpoint
                processed_df_indices.add(df_idx)
                if "sampled_indices" in checkpoint and checkpoint["sampled_indices"]:
                    if df_idx < len(checkpoint["sampled_indices"]):
                        orig_idx = checkpoint["sampled_indices"][df_idx]
                        processed_indices.add(orig_idx)
                else:
                    processed_indices.add(df_idx)
                
                stats["total_processed"] = len(processed_indices)
                checkpoint["processed_indices"] = list(processed_indices)
                checkpoint["last_processed"] = df_idx
                checkpoint.update(stats)
                save_checkpoint(checkpoint)
                
                pbar.update(1)
    
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
