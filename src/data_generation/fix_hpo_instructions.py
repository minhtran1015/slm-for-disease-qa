#!/usr/bin/env python3
"""
Surface-level fix for HPO data: Standardize instruction prefix in existing files.
Converts varied instructions to standardized prefix without regenerating data.
"""

import json
from pathlib import Path
from tqdm import tqdm
import re

STANDARD_INSTRUCTION_PREFIX = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "

# Patterns to detect old HPO instruction formats
OLD_INSTRUCTION_PATTERNS = [
    r"Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai\.",
    r"Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa\.",
    r"Xác định tính đúng sai của nhận định sau về triệu chứng y học\.",
    r"Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau\.",
    r"Dựa vào phân loại triệu chứng y học, hãy trả lời Đúng hoặc Sai\.",
    r"Với kiến thức về bệnh học, hãy xác nhận câu sau Đúng hay Sai\.",
    r"Câu hỏi về mối quan hệ triệu chứng - Trả lời Đúng hoặc Sai\.",
    r"Theo hệ thống phân loại y khoa, câu sau Đúng hay Sai\?",
]

def is_hpo_instruction(instruction: str) -> bool:
    """Check if this is an old HPO instruction that needs fixing."""
    for pattern in OLD_INSTRUCTION_PATTERNS:
        if re.search(pattern, instruction):
            return True
    return False

def extract_question_from_hpo(instruction: str, input_text: str) -> str:
    """Extract the actual question from HPO instruction+input format."""
    
    # If input is empty, the question is probably embedded in instruction
    if not input_text.strip():
        # Find question after the instruction pattern
        for pattern in OLD_INSTRUCTION_PATTERNS:
            match = re.search(pattern + r"\s*(.*)", instruction)
            if match:
                return match.group(1).strip()
        
        # Fallback: take everything after the first sentence
        sentences = instruction.split('.')
        if len(sentences) > 1:
            return '.'.join(sentences[1:]).strip()
        
        return instruction
    else:
        # Question is in input field
        return input_text.strip()

def fix_hpo_format(input_file: str, output_file: str) -> tuple[int, int]:
    """Fix HPO instruction format in existing file."""
    
    fixed = 0
    unchanged = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Fixing {Path(input_file).name}"):
            try:
                sample = json.loads(line)
                
                instruction = sample.get('instruction', '').strip()
                input_text = sample.get('input', '').strip()
                output = sample.get('output', '').strip()
                
                # Check if this needs HPO instruction fixing
                if is_hpo_instruction(instruction):
                    # Extract the actual question
                    question = extract_question_from_hpo(instruction, input_text)
                    
                    if question:
                        # Create standardized format
                        sample['instruction'] = STANDARD_INSTRUCTION_PREFIX + question
                        sample['input'] = ""
                        fixed += 1
                    else:
                        unchanged += 1
                else:
                    unchanged += 1
                
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    
            except Exception as e:
                print(f"⚠️ Error processing line: {e}")
                f_out.write(line)  # Write original line if error
                unchanged += 1
                continue
    
    return fixed, unchanged

def main():
    """Fix HPO instruction format in existing files."""
    
    files_to_fix = [
        ("HPO/hpo_vietnamese_bilingual_train.jsonl", "HPO/hpo_vietnamese_bilingual_train_fixed.jsonl"),
        ("HPO/hpo_vietnamese_bilingual_test.jsonl", "HPO/hpo_vietnamese_bilingual_test_fixed.jsonl"),
    ]
    
    print("🔧 Fixing HPO instruction format to standardized prefix...\n")
    
    total_fixed = 0
    total_unchanged = 0
    
    for input_file, output_file in files_to_fix:
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            continue
        
        print(f"📝 Fixing: {input_file}")
        fixed, unchanged = fix_hpo_format(str(input_path), str(output_path))
        
        total_fixed += fixed
        total_unchanged += unchanged
        
        print(f"   🔧 Fixed: {fixed}")
        print(f"   ✅ Unchanged: {unchanged}")
        print(f"   💾 Output: {output_file}\n")
    
    print("=" * 60)
    print("📊 HPO FIX SUMMARY")
    print("=" * 60)
    print(f"Total fixed: {total_fixed}")
    print(f"Total unchanged: {total_unchanged}")
    print("\n✅ HPO instruction format fixed!")
    print("📝 Now re-run unify_training_data.py with updated file paths.")

if __name__ == "__main__":
    main()