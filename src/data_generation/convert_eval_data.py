#!/usr/bin/env python3
"""
Convert dev_v0.5.csv and test_v0.5.csv to JSONL format
========================================================
Converts evaluation CSV files to Gemma chat format:
- TRUE -> "Đúng"
- FALSE -> "Sai"

Usage: python convert_eval_data.py
Output: 
  - train_data/val.jsonl (from dev_v0.5.csv)
  - train_data/test.jsonl (from test_v0.5.csv)
"""

import csv
import json
import os
from pathlib import Path

OUTPUT_DIR = "../train_data"

# Standardized instruction prefix (matching train.jsonl format)
INSTRUCTION_PREFIX = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "

# CSV files to process
CSV_FILES = [
    ("../data/dev_v0.5.csv", "val.jsonl"),
    ("../data/test_v0.5.csv", "test.jsonl"),
]

def normalize_answer(answer_str):
    """Convert TRUE/FALSE to Đúng/Sai"""
    ans = answer_str.strip().upper()
    if ans in ['TRUE', 'ĐÚNG']:
        return "Đúng"
    elif ans in ['FALSE', 'SAI']:
        return "Sai"
    else:
        # Try to infer
        if ans.startswith('T'):
            return "Đúng"
        else:
            return "Sai"

def process_csv(csv_path, output_jsonl_path):
    """Convert CSV to JSONL format"""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Read CSV with semicolon delimiter
        reader = csv.DictReader(f, delimiter=';')
        
        for row_num, row in enumerate(reader, 1):
            try:
                # Extract question and answer
                # Column format: STT;Mệnh đề Câu hỏi;Đáp án
                question = row.get('Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)', '').strip()
                answer_str = row.get('Đáp án (TRUE/FALSE)', '').strip()
                
                if not question or not answer_str:
                    continue
                
                # Normalize answer
                final_answer = normalize_answer(answer_str)
                
                # Add standardized instruction prefix to question
                full_question = INSTRUCTION_PREFIX + question
                
                # Create Gemma chat format entry
                entry = {
                    "messages": [
                        {"role": "user", "content": full_question},
                        {"role": "model", "content": final_answer}
                    ]
                }
                
                data.append(entry)
                
            except Exception as e:
                print(f"  ⚠️  Skipped row {row_num}: {str(e)}")
                continue
    
    # Save
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(data)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("=" * 70)
print("📊 EVALUATION DATA CONVERSION")
print("=" * 70)

for csv_file, jsonl_file in CSV_FILES:
    csv_path = csv_file
    output_path = os.path.join(OUTPUT_DIR, jsonl_file)
    
    if not os.path.exists(csv_path):
        print(f"\n⚠️  SKIPPED (not found): {csv_path}")
        continue
    
    print(f"\n📂 Converting: {csv_path}")
    print(f"   Output: {output_path}")
    
    count = process_csv(csv_path, output_path)
    print(f"   ✅ {count:,} samples converted")

print("\n" + "=" * 70)
print("✨ EVALUATION DATA READY!")
print("=" * 70)
print(f"✓ Dev → {os.path.join(OUTPUT_DIR, 'val.jsonl')}")
print(f"✓ Test → {os.path.join(OUTPUT_DIR, 'test.jsonl')}")
print("✓ All answers normalized to 'Đúng/Sai'")
print("✓ All formats unified to Gemma 'messages' style")
print()
