#!/usr/bin/env python3
"""
Create train_data_v2 Dataset
============================
- Validation: Extract from ViMedAQA only (for domain-aligned evaluation)
- Training: Mix ICD, HPO, CLTL, DrugBank, ViMedAQA with ViMedAQA oversampling
- Output: Gemma chat format (user/model roles)

Usage: python create_train_data_v2.py
Output: train_data_v2/train.jsonl, train_data_v2/val.jsonl
"""

import json
import random
import os
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_DIR = "../train_data_v2"
RANDOM_SEED = 42

# ViMedAQA oversampling factor (2x to emphasize Vietnamese yes/no QA)
VIMEDAQA_OVERSAMPLE_FACTOR = 2

# Validation set: 15% of ViMedAQA train data
VIMEDAQA_VAL_RATIO = 0.15

# Source files (all in alpaca format: instruction/input/output)
FILES = {
    "vimedaqa": "../ViMedAQA/vimedaqa_yesno_train_split_standardized.jsonl",
    "drugbank": "../DrugBank/drugbank_qa_vietnamese_20k.jsonl",
    "hpo": "../HPO/hpo_vietnamese_bilingual_train_fixed.jsonl",
    "icd10": "../ICD10/generated_questions/icd10_yesno_balanced.jsonl",
    "cltl": "../data/medical_qa_vietnamese_cltl_train_fixed.jsonl",
}


def clean_answer(answer: str) -> str:
    """Normalize answer to Đúng/Sai."""
    a = answer.strip().lower()
    if a in ['có', 'yes', 'true', 'đúng']:
        return "Đúng"
    if a in ['không', 'no', 'false', 'sai']:
        return "Sai"
    return "Đúng"  # Fallback


def load_alpaca_file(fpath: str) -> list:
    """Load alpaca format file and convert to chat format."""
    data = []
    
    if not os.path.exists(fpath):
        print(f"⚠️  File not found: {fpath}")
        return data
    
    with open(fpath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                
                # Extract instruction and input
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
                final_answer = clean_answer(raw_answer)
                
                # Convert to Gemma chat format
                entry = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "model", "content": final_answer}
                    ]
                }
                
                data.append(entry)
                
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    return data


def main():
    print("=" * 70)
    print("🔄 CREATE TRAIN_DATA_V2 PIPELINE")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    print(f"📈 ViMedAQA oversampling factor: {VIMEDAQA_OVERSAMPLE_FACTOR}x")
    print(f"📊 ViMedAQA validation ratio: {VIMEDAQA_VAL_RATIO * 100}%")
    print()
    
    random.seed(RANDOM_SEED)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    stats = {"sources": {}, "oversampled": {}}
    
    # --- Step 1: Load ViMedAQA and split for validation ---
    print("📂 Loading ViMedAQA for validation extraction...")
    vimedaqa_data = load_alpaca_file(FILES["vimedaqa"])
    
    # Shuffle before splitting
    random.shuffle(vimedaqa_data)
    
    # Split into validation and training
    val_size = int(len(vimedaqa_data) * VIMEDAQA_VAL_RATIO)
    vimedaqa_val = vimedaqa_data[:val_size]
    vimedaqa_train = vimedaqa_data[val_size:]
    
    print(f"   ✅ Total ViMedAQA: {len(vimedaqa_data)}")
    print(f"   📋 Validation set: {len(vimedaqa_val)}")
    print(f"   📋 Training set: {len(vimedaqa_train)}")
    
    stats["sources"]["vimedaqa_original"] = len(vimedaqa_train)
    stats["vimedaqa_val"] = len(vimedaqa_val)
    
    # --- Step 2: Apply ViMedAQA oversampling ---
    print(f"\n📈 Oversampling ViMedAQA by {VIMEDAQA_OVERSAMPLE_FACTOR}x...")
    vimedaqa_oversampled = vimedaqa_train * VIMEDAQA_OVERSAMPLE_FACTOR
    print(f"   ✅ After oversampling: {len(vimedaqa_oversampled)}")
    stats["oversampled"]["vimedaqa"] = len(vimedaqa_oversampled)
    
    # --- Step 3: Load other datasets ---
    train_data = list(vimedaqa_oversampled)  # Start with oversampled ViMedAQA
    
    for name, fpath in FILES.items():
        if name == "vimedaqa":
            continue  # Already processed
        
        print(f"\n📂 Loading {name}...")
        data = load_alpaca_file(fpath)
        train_data.extend(data)
        stats["sources"][name] = len(data)
        print(f"   ✅ {len(data)} samples")
    
    # --- Step 4: Shuffle training data ---
    print("\n🔀 Shuffling training data...")
    random.shuffle(train_data)
    
    # --- Step 5: Save files ---
    train_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_file = os.path.join(OUTPUT_DIR, "val.jsonl")
    stats_file = os.path.join(OUTPUT_DIR, "stats.json")
    
    print(f"\n💾 Saving training data to: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 Saving validation data to: {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in vimedaqa_val:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Calculate final stats
    stats["total_train"] = len(train_data)
    stats["total_val"] = len(vimedaqa_val)
    stats["vimedaqa_percentage"] = round(
        len(vimedaqa_oversampled) / len(train_data) * 100, 2
    )
    
    print(f"💾 Saving stats to: {stats_file}")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("📊 TRAIN_DATA_V2 SUMMARY")
    print("=" * 70)
    print("\nSource breakdown:")
    for name, count in stats["sources"].items():
        print(f"  {name}: {count:,}")
    
    print(f"\nAfter ViMedAQA {VIMEDAQA_OVERSAMPLE_FACTOR}x oversampling:")
    print(f"  ViMedAQA: {stats['oversampled']['vimedaqa']:,} samples")
    
    print(f"\n{'TOTAL TRAINING':.<50} {stats['total_train']:>10,}")
    print(f"{'TOTAL VALIDATION (ViMedAQA only)':.<50} {stats['total_val']:>10,}")
    print(f"{'ViMedAQA % of training':.<50} {stats['vimedaqa_percentage']:>10}%")
    
    print("\n" + "=" * 70)
    print("✨ TRAIN_DATA_V2 READY!")
    print("=" * 70)
    print("✓ Validation set: ViMedAQA only (domain-aligned evaluation)")
    print("✓ Training set: Mixed all sources with ViMedAQA oversampling")
    print("✓ Format: Gemma chat (user/model roles)")
    print(f"✓ Output: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
