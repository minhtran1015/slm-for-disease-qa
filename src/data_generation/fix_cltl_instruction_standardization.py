#!/usr/bin/env python3
"""
Fix CLTL Instruction Standardization
====================================
Standardizes all CLTL instruction formats to use the consistent prefix.
This fixes the instruction inconsistency issue that causes models to ignore instructions.
"""

import json
import os

def fix_cltl_instructions():
    """Fix CLTL instruction formats to use standardized prefix."""
    
    # Standardized instruction format (same as other datasets)
    STANDARD_INSTRUCTION = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "
    
    # All inconsistent CLTL instruction formats to replace
    INCONSISTENT_INSTRUCTIONS = [
        "Áp dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt. Trả lời Đúng hoặc Sai.",
        "Chuyển giao tri thức y khoa quốc tế sang tư duy tiếng Việt. Trả lời Đúng hoặc Sai.",
        "Dựa trên bằng chứng y khoa quốc tế, đánh giá câu hỏi tiếng Việt và trả lời Đúng hoặc Sai.",
        "Dựa vào bối cảnh y khoa quốc tế, hãy trả lời câu hỏi bằng tiếng Việt. Trả lời Đúng hoặc Sai.",
        "Kết hợp kiến thức y học quốc tế với ngôn ngữ Việt Nam để trả lời. Chọn Đúng hoặc Sai.",
        "Phân tích thông tin y khoa từ tài liệu quốc tế và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai.",
        "Sử dụng cơ sở dữ liệu y khoa quốc tế để phân tích và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai.",
        "Sử dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt. Chỉ trả lời Đúng hoặc Sai."
    ]
    
    input_file = "../data/medical_qa_vietnamese_cltl_train.jsonl"
    output_file = "../data/medical_qa_vietnamese_cltl_train_fixed.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    fixed_count = 0
    total_count = 0
    
    print("🔧 Fixing CLTL instruction standardization...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1
                
                # Check if instruction needs fixing
                current_instruction = data.get('instruction', '')
                if current_instruction in INCONSISTENT_INSTRUCTIONS:
                    # Replace with standardized instruction + input
                    input_text = data.get('input', '')
                    standardized_instruction = STANDARD_INSTRUCTION + input_text
                    
                    # Update the data format
                    data['instruction'] = standardized_instruction
                    data['input'] = ""  # Clear input since it's now in instruction
                    fixed_count += 1
                    
                    if fixed_count <= 3:  # Log first few fixes
                        print(f"  Fix {fixed_count}: '{current_instruction[:50]}...' → standardized")
                
                # Write the (potentially fixed) data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"❌ Error on line {line_num}: {e}")
                continue
    
    print(f"\n✅ CLTL instruction standardization completed!")
    print(f"  Total samples: {total_count}")
    print(f"  Fixed samples: {fixed_count}")
    print(f"  Output file: {output_file}")
    
    return True

def main():
    """Main function to fix CLTL instructions."""
    print("=" * 70)
    print("🔧 CLTL INSTRUCTION STANDARDIZATION FIX")
    print("=" * 70)
    print("Problem: CLTL data has 8 different instruction formats")
    print("Solution: Standardize to consistent prefix")
    print("=" * 70)
    
    success = fix_cltl_instructions()
    
    if success:
        print("\n🎉 Ready for re-unification with consistent instructions!")
        print("\nNext steps:")
        print("1. Update unify_training_data_v2.py to use the fixed CLTL file")
        print("2. Re-run the unification to generate consistent train_v2.jsonl")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")

if __name__ == "__main__":
    main()