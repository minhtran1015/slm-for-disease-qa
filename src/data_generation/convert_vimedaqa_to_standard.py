#!/usr/bin/env python3
"""
Surface-level fix for ViMedAQA: Convert chat format to standardized instruction/input/output format.
Does NOT regenerate data, just converts existing files to our standard format.
"""

import json
from pathlib import Path
from tqdm import tqdm

STANDARD_INSTRUCTION_PREFIX = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "

def convert_vimedaqa_format(input_file: str, output_file: str) -> tuple[int, int]:
    """Convert ViMedAQA chat format to standardized format."""
    
    converted = 0
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Converting {Path(input_file).name}"):
            try:
                sample = json.loads(line)
                
                # Extract answer from standardized output format
                if "messages" in sample and len(sample["messages"]) >= 3:
                    # Chat format: system -> user -> assistant
                    user_msg = sample["messages"][1].get("content", "").strip()
                    assistant_msg = sample["messages"][2].get("content", "").strip()
                    
                    if user_msg and assistant_msg in ["Đúng", "Sai"]:
                        # Create standardized format
                        standardized = {
                            "instruction": STANDARD_INSTRUCTION_PREFIX + user_msg,
                            "input": "",
                            "output": assistant_msg,
                            # Keep metadata
                            "question_type": sample.get("question_type", ""),
                            "statement_id": sample.get("statement_id", ""),
                            "source": sample.get("source", "vimedaqa")
                        }
                        
                        f_out.write(json.dumps(standardized, ensure_ascii=False) + '\n')
                        converted += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing line: {e}")
                skipped += 1
                continue
    
    return converted, skipped

def main():
    """Convert all ViMedAQA files to standardized format."""
    
    base_dir = Path("ViMedAQA")
    
    files_to_convert = [
        ("vimedaqa_yesno_train_split.jsonl", "vimedaqa_yesno_train_split_standardized.jsonl"),
        ("vimedaqa_yesno_test_split.jsonl", "vimedaqa_yesno_test_split_standardized.jsonl"),
    ]
    
    print("🔄 Converting ViMedAQA from chat format to standardized format...\n")
    
    total_converted = 0
    total_skipped = 0
    
    for input_file, output_file in files_to_convert:
        input_path = base_dir / input_file
        output_path = base_dir / output_file
        
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            continue
        
        print(f"📝 Converting: {input_file}")
        converted, skipped = convert_vimedaqa_format(str(input_path), str(output_path))
        
        total_converted += converted
        total_skipped += skipped
        
        print(f"   ✅ Converted: {converted}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   💾 Output: {output_file}\n")
    
    print("=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total converted: {total_converted}")
    print(f"Total skipped: {total_skipped}")
    print("\n✅ Conversion complete!")
    print("📝 Standardized files are ready for merging into unified pipeline.")

if __name__ == "__main__":
    main()
