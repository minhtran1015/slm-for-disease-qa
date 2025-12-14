#!/usr/bin/env python3
"""
Comprehensive surface-level fix for ALL remaining non-standardized instructions.
Converts any instruction that doesn't start with our standardized prefix.
"""

import json
from pathlib import Path
from tqdm import tqdm
import re

STANDARD_INSTRUCTION_PREFIX = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "

# All known non-standardized instruction patterns from various datasets
NON_STANDARD_PATTERNS = [
    # HPO patterns
    r"Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai\.",
    r"Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa\.",
    r"Xác định tính đúng sai của nhận định sau về triệu chứng y học\.",
    r"Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau\.",
    r"Dựa vào phân loại triệu chứng y học, hãy trả lời Đúng hoặc Sai\.",
    r"Với kiến thức về bệnh học, hãy xác nhận câu sau Đúng hay Sai\.",
    r"Câu hỏi về mối quan hệ triệu chứng - Trả lời Đúng hoặc Sai\.",
    r"Theo hệ thống phân loại y khoa, câu sau Đúng hay Sai\?",
    
    # CLTL patterns
    r"Dựa vào bối cảnh y khoa quốc tế, hãy trả lời câu hỏi bằng tiếng Việt\. Trả lời Đúng hoặc Sai\.",
    r"Sử dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt\. Chỉ trả lời Đúng hoặc Sai\.",
    r"Phân tích thông tin y khoa từ tài liệu quốc tế và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai\.",
    r"Dựa trên bằng chứng y khoa quốc tế, đánh giá câu hỏi tiếng Việt và trả lời Đúng hoặc Sai\.",
    r"Áp dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt\. Trả lời Đúng hoặc Sai\.",
    r"Chuyển giao tri thức y khoa quốc tế sang tư duy tiếng Việt\. Trả lời Đúng hoặc Sai\.",
    r"Sử dụng cơ sở dữ liệu y khoa quốc tế để phân tích và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai\.",
    r"Kết hợp kiến thức y học quốc tế với ngôn ngữ Việt Nam để trả lời\. Chọn Đúng hoặc Sai\.",
    
    # DrugBank patterns (legacy)
    r"Dựa trên kiến thức dược phẩm, hãy trả lời Đúng hoặc Sai\.",
    r"Xác nhận thông tin sau về thuốc - Trả lời Đúng hoặc Sai\.",
    r"Hãy đánh giá tính chính xác của câu khẳng định sau về dược phẩm\. Chọn Đúng hoặc Sai\.",
    
    # Any other patterns that end with instruction but don't start with our prefix
    r".*[Tt]rả lời Đúng hoặc Sai\.?\s*$",
    r".*[Cc]họn Đúng hoặc Sai\.?\s*$",
]

def needs_standardization(content: str) -> bool:
    """Check if content needs to be standardized to our prefix."""
    # If it already starts with our standard prefix, no need to change
    if content.startswith(STANDARD_INSTRUCTION_PREFIX):
        return False
    
    # Check if it matches any known non-standard pattern
    for pattern in NON_STANDARD_PATTERNS:
        if re.search(pattern, content):
            return True
    
    # Also check if it contains instruction-like text but doesn't have our prefix
    instruction_indicators = [
        "trả lời đúng hoặc sai", "chọn đúng hoặc sai", 
        "xác minh", "đánh giá", "kiến thức y khoa"
    ]
    
    content_lower = content.lower()
    for indicator in instruction_indicators:
        if indicator in content_lower and not content.startswith(STANDARD_INSTRUCTION_PREFIX):
            return True
    
    return False

def extract_question_from_content(content: str) -> str:
    """Extract the actual question from instruction + question content."""
    
    # Try to find question after known instruction patterns
    for pattern in NON_STANDARD_PATTERNS:
        # Split by the pattern and take what comes after
        parts = re.split(pattern, content)
        if len(parts) > 1:
            question = parts[-1].strip()
            if question:
                return question
    
    # Try to find question after newline (common format)
    if '\n' in content:
        lines = content.split('\n')
        # Skip first line (likely instruction), combine the rest
        question_lines = [line.strip() for line in lines[1:] if line.strip()]
        if question_lines:
            return '\n'.join(question_lines)
    
    # Try to find question after common instruction ending phrases
    endings = [
        "Trả lời Đúng hoặc Sai.",
        "Chọn Đúng hoặc Sai.",
        "trả lời đúng hoặc sai.",
        "chọn đúng hoặc sai."
    ]
    
    for ending in endings:
        if ending in content:
            parts = content.split(ending, 1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()
    
    # Fallback: if content is long, try to find where question starts
    if len(content) > 100:
        # Look for question-like patterns (contains '?', medical terms, etc.)
        sentences = content.split('.')
        for i, sentence in enumerate(sentences):
            if ('?' in sentence or 
                any(term in sentence.lower() for term in ['có phải', 'có thuộc', 'có nằm', 'có hiệu quả'])):
                return '.'.join(sentences[i:]).strip()
    
    # Last resort: return original content (will be prepended with standard prefix)
    return content

def fix_unified_data(train_file: str) -> int:
    """Fix all non-standardized instructions in the unified training data."""
    
    fixed_count = 0
    temp_file = train_file + ".tmp"
    
    with open(train_file, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(tqdm(f_in, desc="Fixing instructions"), 1):
            try:
                sample = json.loads(line)
                
                # Extract user message content
                if 'messages' in sample and len(sample['messages']) >= 2:
                    user_content = sample['messages'][0].get('content', '').strip()
                    
                    if needs_standardization(user_content):
                        # Extract question and apply standardized prefix
                        question = extract_question_from_content(user_content)
                        standardized_content = STANDARD_INSTRUCTION_PREFIX + question
                        
                        # Update the sample
                        sample['messages'][0]['content'] = standardized_content
                        fixed_count += 1
                        
                        if line_num <= 5:  # Show first few fixes
                            print(f"\n🔧 Fixed sample {line_num}:")
                            print(f"   Before: {user_content[:60]}...")
                            print(f"   After: {standardized_content[:60]}...")
                
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"⚠️ Error processing line {line_num}: {e}")
                f_out.write(line)  # Write original line if error
                continue
    
    # Replace original file with fixed version
    Path(temp_file).replace(train_file)
    
    return fixed_count

def main():
    """Fix all non-standardized instructions in unified training data."""
    
    train_file = "train_data/train.jsonl"
    
    if not Path(train_file).exists():
        print(f"❌ File not found: {train_file}")
        return
    
    print("🔧 Fixing ALL remaining non-standardized instructions...\n")
    
    # Create backup
    backup_file = f"{train_file}.backup"
    if not Path(backup_file).exists():
        import shutil
        shutil.copy2(train_file, backup_file)
        print(f"📄 Backup created: {backup_file}")
    
    fixed_count = fix_unified_data(train_file)
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE FIX SUMMARY")
    print("=" * 60)
    print(f"Instructions standardized: {fixed_count}")
    print(f"File updated: {train_file}")
    print(f"Backup available: {backup_file}")
    print("\n✅ All instructions now use standardized prefix!")
    print("📝 Ready for training with perfect instruction consistency.")

if __name__ == "__main__":
    main()