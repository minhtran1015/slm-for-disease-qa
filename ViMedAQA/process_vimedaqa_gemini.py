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
MAX_SAMPLES = 10000  # Hardcoded to process 10k samples (produces ~20k outputs)
BALANCE_RATIO = 1.0  # 1.0 = equal Đúng/Sai samples

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
    
    # Configure generation settings
    generation_config = genai.GenerationConfig(
        temperature=0.3,  # Lower temperature for consistent outputs
        max_output_tokens=500,
        top_p=0.9,
    )
    
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=generation_config,
    )
    
    return model


def create_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """
    Create a prompt for Gemini to transform Q&A into a True/False statement.
    
    The goal is to generate statements like:
    - "Ho kéo dài trên 3 tuần có phải là triệu chứng của lao phổi."
    - "Sỏi thận hình thành do khoáng chất kết tụ trong nước tiểu."
    """
    prompt = f"""Bạn là chuyên gia y khoa Việt Nam. Nhiệm vụ của bạn là chuyển đổi cặp câu hỏi-trả lời thành một câu khẳng định y khoa để kiểm tra kiến thức Đúng/Sai.

**Quy tắc:**
1. Tạo MỘT câu khẳng định ngắn gọn, rõ ràng bằng tiếng Việt
2. Câu khẳng định phải mang tính y khoa chuyên môn
3. Câu phải có thể trả lời Đúng hoặc Sai một cách rõ ràng
4. KHÔNG thêm "(Đ/S)" vào cuối câu
5. Giữ nguyên thuật ngữ y khoa quan trọng
6. Câu phải tự nhiên và súc tích

**Ví dụ đầu vào:**
Câu hỏi: Paracetamol có phải là một loại thuốc giảm đau không?
Trả lời: Có, Paracetamol là thuốc giảm đau hạ sốt phổ biến.

**Ví dụ đầu ra mong muốn:**
Paracetamol là một loại thuốc có tác dụng giảm đau và hạ sốt.

---

**Câu hỏi:** {question}

**Trả lời:** {answer}

{f"**Ngữ cảnh bổ sung:** {context}" if context else ""}

**Câu khẳng định y khoa (chỉ trả lời câu khẳng định, không giải thích):**"""
    
    return prompt


def create_false_statement_prompt(question: str, answer: str, context: str = "") -> str:
    """
    Create a prompt for Gemini to generate a FALSE medical statement.
    This creates a statement that sounds plausible but is medically incorrect.
    """
    prompt = f"""Bạn là chuyên gia y khoa Việt Nam. Nhiệm vụ của bạn là tạo một câu khẳng định y khoa SAI (nhưng có vẻ hợp lý) dựa trên thông tin dưới đây.

**Quy tắc:**
1. Tạo MỘT câu khẳng định SAI về mặt y khoa
2. Câu phải có vẻ hợp lý để kiểm tra kiến thức y khoa
3. Thay đổi một chi tiết quan trọng để câu trở thành SAI (ví dụ: thay đổi liều lượng, công dụng, cách dùng, tác dụng phụ)
4. KHÔNG thêm "(Đ/S)" vào cuối câu
5. Câu phải tự nhiên và súc tích
6. SAI một cách tinh vi, không quá rõ ràng

**Ví dụ đầu vào:**
Câu hỏi: Paracetamol có phải là một loại thuốc giảm đau không?
Trả lời: Có, Paracetamol là thuốc giảm đau hạ sốt phổ biến.

**Ví dụ đầu ra mong muốn (câu SAI):**
Paracetamol là thuốc kháng sinh điều trị nhiễm khuẩn.

---

**Câu hỏi:** {question}

**Trả lời:** {answer}

{f"**Ngữ cảnh bổ sung:** {context}" if context else ""}

**Câu khẳng định SAI về y khoa (chỉ trả lời câu khẳng định, không giải thích):**"""
    
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
            
            # Check if response has text
            if response and response.text:
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
    Process a batch of samples with parallel API calls.
    """
    results = []
    
    # Create all prompts first
    prompts = []
    sample_info = []
    
    for i, row in enumerate(batch_samples):
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        context = str(row.get('context', '')).strip() if 'context' in row else ""
        
        if not question or not answer:
            continue
            
        # True statement prompt
        true_prompt = create_statement_prompt(question, answer, context)
        false_prompt = create_false_statement_prompt(question, answer, context)
        
        prompts.extend([true_prompt, false_prompt])
        sample_info.extend([
            (question, answer, True, batch_start_idx + i),
            (question, answer, False, batch_start_idx + i)
        ])
    
    # Process prompts with ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all API calls
        future_to_info = {}
        for i, prompt in enumerate(prompts):
            future = executor.submit(call_gemini_api, model, prompt)
            future_to_info[future] = sample_info[i]
            
            # Small delay between submissions for rate limiting
            time.sleep(60 / REQUESTS_PER_MINUTE / 10)  # Distributed delay
        
        # Collect results as they complete
        for future in as_completed(future_to_info):
            question, answer, is_true, sample_idx = future_to_info[future]
            
            try:
                statement = future.result()
                
                if statement:
                    # Clean the statement
                    statement = statement.replace("(Đ/S)", "").strip().rstrip('.')
                    
                    result = {
                        "instruction": random.choice(INSTRUCTION_TEMPLATES),
                        "input": statement,
                        "output": "Đúng" if is_true else "Sai",
                        "question_type": "vimedaqa_true" if is_true else "vimedaqa_false",
                        "source_question": question,
                        "source_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                        "sample_idx": sample_idx
                    }
                    results.append(result)
                    
            except Exception as e:
                continue  # Skip failed samples
    
    return results


def main():
    """Main processing pipeline - Hardcoded to generate 20k balanced samples."""
    
    print("=" * 60)
    print("🚀 ViMedAQA Gemini Processing Pipeline - 20k Output Generation")
    print("=" * 60)
    print(f"   Processing: {MAX_SAMPLES:,} samples → ~{MAX_SAMPLES*2:,} outputs")
    print(f"   Model: {GEMINI_MODEL}")
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
                        for result in batch_results:
                            sample_idx = result.pop('sample_idx', 0)
                            processed_df_indices.add(sample_idx)
                            
                            # Map back to original index for checkpoint
                            if "sampled_indices" in checkpoint and checkpoint["sampled_indices"] is not None:
                                if sample_idx < len(checkpoint["sampled_indices"]):
                                    orig_idx = checkpoint["sampled_indices"][sample_idx]
                                    processed_indices.add(orig_idx)
                            else:
                                processed_indices.add(sample_idx)
                            
                            # Update stats
                            if result["output"] == "Đúng":
                                stats["successful_true"] += 1
                            else:
                                stats["successful_false"] += 1
                        
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
    
    print("\n" + "=" * 60)
    print("📊 Processing Complete!")
    print("=" * 60)
    print(f"   Total samples processed: {stats['total_processed']}")
    print(f"   Successful TRUE statements: {stats['successful_true']}")
    print(f"   Successful FALSE statements: {stats['successful_false']}")
    print(f"   Failed: {stats['failed']}")
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
                if i >= 3:
                    break
                sample = json.loads(line)
                print(f"\n   [{i+1}] {sample['input']}")
                print(f"       → {sample['output']}")


if __name__ == "__main__":
    main()
