#!/usr/bin/env python3
"""
Unified Training Data Pipeline V2
==================================
Merges all training files (DrugBank, HPO, CLTL, ICD-10, ViMedAQA v2) into:
1. Single train_v2.jsonl with Gemma chat format (user/model roles)
2. Consistent "Đúng/Sai" answers across all datasets
3. Uses the enhanced v2 ViMedAQA dataset (40k samples)
4. Shuffled to mix reasoning with knowledge

Usage: python unify_training_data_v2.py
Output: train_data/train_v2.jsonl (ready for Gemma 1B-IT training)
"""

import json
import random
import os
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = "../train_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_v2.jsonl")

# Mapping specific file paths - ALL with standardized instruction prefix
# Using v2 ViMedAQA dataset with 40k samples instead of original ~18k
FILES = [
    # (Path, Type, Fix_Co_Khong_Flag)
    # DrugBank - using standardized format
    ("../DrugBank/drugbank_qa_vietnamese_20k.jsonl", "alpaca", False),
    # HPO - FIXED instruction format to use standardized prefix
    ("../HPO/hpo_vietnamese_bilingual_train_fixed.jsonl", "alpaca", False),
    # CLTL - using existing (already in correct format)
    ("../data/medical_qa_vietnamese_cltl_train.jsonl", "alpaca", False),
    # ICD-10 - using standardized prefix and Đúng/Sai
    ("../ICD10/generated_questions/icd10_yesno_full.jsonl", "alpaca", False),
    # ViMedAQA V2 - using the enhanced v2 dataset with 40k samples
    ("../ViMedAQA/vimedaqa_yesno_train_v2_standardized.jsonl", "alpaca", False),
]

def clean_answer(answer, needs_fix=False):
    """
    Normalize all variations to standard Vietnamese Yes/No.
    
    Maps:
    - "Có", "Yes", "True", "Đúng" -> "Đúng"
    - "Không", "No", "False", "Sai" -> "Sai"
    """
    a = answer.strip().lower()
    
    # Handle "Có/Không" specifically (ICD-10 legacy format)
    if needs_fix:
        if a in ['có', 'yes', 'true', 'đúng']: 
            return "Đúng"
        if a in ['không', 'no', 'false', 'sai']: 
            return "Sai"
    else:
        # Standard format
        if a in ['có', 'yes', 'true', 'đúng']: 
            return "Đúng"
        if a in ['không', 'no', 'false', 'sai']: 
            return "Sai"
    
    return "Đúng"  # Fallback for safety

def process_file(fpath, ftype, needs_fix):
    """Process a single file and yield unified entries."""
    data = []
    
    with open(fpath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                
                # 1. Extract question and answer based on format
                if ftype == "alpaca":
                    # Standard format: instruction + input -> output
                    instruction = row.get('instruction', '').strip()
                    input_text = row.get('input', '').strip()
                    
                    # Combine instruction and input as question
                    if instruction and input_text:
                        question = f"{instruction}\n{input_text}"
                    elif input_text:
                        question = input_text
                    else:
                        question = instruction
                    
                    raw_answer = row.get('output', '').strip()
                    
                elif ftype == "chat":
                    # Chat format: extract from messages array
                    messages = row.get('messages', [])
                    if len(messages) < 2:
                        continue
                    
                    # Find last user message as question
                    question = None
                    for msg in reversed(messages):
                        if msg.get('role') == 'user':
                            question = msg.get('content', '').strip()
                            break
                    
                    if not question:
                        continue
                    
                    # Find assistant response as answer
                    raw_answer = None
                    for msg in reversed(messages):
                        if msg.get('role') in ['assistant', 'model']:
                            raw_answer = msg.get('content', '').strip()
                            break
                    
                    if not raw_answer:
                        continue
                else:
                    continue
                
                # 2. Normalize answer to "Đúng/Sai"
                final_answer = clean_answer(raw_answer, needs_fix)
                
                # 3. Format for Gemma chat model
                # Gemma 1B-IT expects: [user message, model response]
                entry = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "model", "content": final_answer}
                    ]
                }
                
                data.append(entry)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    return data


# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

unified_data = []
stats = {}

print("=" * 70)
print("🔄 UNIFIED TRAINING DATA PIPELINE V2")
print("=" * 70)
print("📈 Enhanced with ViMedAQA v2 dataset (40k samples)")
print("=" * 70)

# Process each file
for fpath, ftype, needs_fix in FILES:
    if not os.path.exists(fpath):
        print(f"⚠️  SKIPPED (not found): {fpath}")
        stats[fpath] = 0
        continue
    
    print(f"\n📂 Processing: {fpath}")
    print(f"   Format: {ftype}, Fix Có/Không: {needs_fix}")
    
    file_data = process_file(fpath, ftype, needs_fix)
    unified_data.extend(file_data)
    stats[fpath] = len(file_data)
    
    print(f"   ✅ {len(file_data)} samples extracted")

# Shuffle with fixed seed for reproducibility (different seed for v2)
print("\n🔀 Shuffling data (seed=42v2)...")
random.seed(42)  # Using same seed to allow comparison between v1 and v2
random.shuffle(unified_data)

# Save unified training file
print(f"\n💾 Writing to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in unified_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Calculate answer distribution
print("\n🔍 Analyzing answer distribution...")
answer_counts = {"Đúng": 0, "Sai": 0}
for item in unified_data:
    answer = item["messages"][1]["content"]
    if answer in answer_counts:
        answer_counts[answer] += 1

# Print statistics
print("\n" + "=" * 70)
print("📊 UNIFICATION V2 SUMMARY")
print("=" * 70)
total = 0
for fpath, count in stats.items():
    dataset_name = fpath.split('/')[-1].replace('.jsonl', '')
    print(f"{dataset_name}: {count:,} samples")
    total += count

print(f"\n{'TOTAL TRAINING SAMPLES':.<50} {total:>10,}")
print(f"{'Đúng answers':.<50} {answer_counts['Đúng']:>10,}")
print(f"{'Sai answers':.<50} {answer_counts['Sai']:>10,}")
print(f"{'Balance ratio (Đúng/Sai)':.<50} {answer_counts['Đúng']/answer_counts['Sai']:>10.2f}")
print(f"{'OUTPUT FILE':.<50} {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("✨ TRAINING V2 READY!")
print("=" * 70)
print("✓ All answers normalized to 'Đúng/Sai'")
print("✓ All formats unified to Gemma 'messages' style")
print("✓ Enhanced ViMedAQA v2 dataset included (40k samples)")
print("✓ Data shuffled for optimal learning")
print(f"✓ Ready for SFT training with Gemma 1B-IT")
print(f"✓ Expected ~50% larger ViMedAQA contribution vs v1")
print()
