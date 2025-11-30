#!/usr/bin/env python3
"""
Validation Script for Unified Training Data
============================================
Verifies that all generated JSONL files are correctly formatted
and ready for Gemma 1B-IT training.

Usage: python3 validate_unified_data.py
"""

import json
import os
from pathlib import Path

def validate_file(filepath, expected_role_sequence=["user", "model"]):
    """Validate a single JSONL file"""
    print(f"\n📂 Validating: {filepath}")
    print("─" * 70)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    valid_count = 0
    invalid_count = 0
    answer_distribution = {"Đúng": 0, "Sai": 0, "Other": 0}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # Check structure
                if "messages" not in data:
                    print(f"❌ Line {line_num}: Missing 'messages' key")
                    invalid_count += 1
                    continue
                
                messages = data["messages"]
                
                # Check message roles
                if len(messages) < 2:
                    print(f"❌ Line {line_num}: Less than 2 messages")
                    invalid_count += 1
                    continue
                
                # Verify user/model sequence
                if len(messages) >= 2:
                    roles = [msg.get("role") for msg in messages]
                    if roles[-2] != "user" or roles[-1] != "model":
                        print(f"❌ Line {line_num}: Invalid role sequence: {roles}")
                        invalid_count += 1
                        continue
                
                # Check answer validity
                answer = messages[-1].get("content", "").strip()
                if answer == "Đúng":
                    answer_distribution["Đúng"] += 1
                elif answer == "Sai":
                    answer_distribution["Sai"] += 1
                else:
                    print(f"❌ Line {line_num}: Invalid answer: {answer}")
                    answer_distribution["Other"] += 1
                    invalid_count += 1
                    continue
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: JSON decode error: {str(e)}")
                invalid_count += 1
            except Exception as e:
                print(f"❌ Line {line_num}: {str(e)}")
                invalid_count += 1
    
    total = valid_count + invalid_count
    
    # Print statistics
    print(f"\n✅ Valid lines: {valid_count:,}")
    print(f"❌ Invalid lines: {invalid_count:,}")
    print(f"📊 Total lines: {total:,}")
    
    if answer_distribution["Đúng"] + answer_distribution["Sai"] > 0:
        total_answers = answer_distribution["Đúng"] + answer_distribution["Sai"]
        dung_pct = answer_distribution["Đúng"] * 100 / total_answers
        sai_pct = answer_distribution["Sai"] * 100 / total_answers
        balance = min(answer_distribution["Đúng"], answer_distribution["Sai"]) / max(answer_distribution["Đúng"], answer_distribution["Sai"]) * 100
        
        print(f"\n📊 Answer Distribution:")
        print(f"   Đúng: {answer_distribution['Đúng']:,} ({dung_pct:.1f}%)")
        print(f"   Sai:  {answer_distribution['Sai']:,} ({sai_pct:.1f}%)")
        print(f"   Balance: {balance:.1f}% ✅" if balance > 85 else f"   Balance: {balance:.1f}% ⚠️")
    
    if answer_distribution["Other"] > 0:
        print(f"❌ Invalid answers: {answer_distribution['Other']}")
    
    success = invalid_count == 0 and answer_distribution["Other"] == 0
    return success

def main():
    print("=" * 70)
    print("🔍 UNIFIED TRAINING DATA VALIDATION")
    print("=" * 70)
    
    files_to_validate = [
        ("../train_data/train.jsonl", "Training Data"),
        ("../train_data/val.jsonl", "Validation Data"),
        ("../train_data/test.jsonl", "Test Data"),
    ]
    
    results = {}
    all_valid = True
    
    for filepath, label in files_to_validate:
        is_valid = validate_file(filepath)
        results[label] = is_valid
        all_valid = all_valid and is_valid
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    for label, is_valid in results.items():
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{label}:".ljust(25) + f" {status}")
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✨ ALL VALIDATIONS PASSED! Data is ready for training. ✨")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Load data: from datasets import load_dataset")
        print("2. Train: Gemma 1B-IT with your preferred framework")
        print("3. Monitor: eval_loss (target: <0.4 for 80% accuracy)")
        print("4. Evaluate: accuracy on test.jsonl")
        return 0
    else:
        print("❌ VALIDATION FAILED! Please check errors above.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())
