#!/usr/bin/env python3
"""
Phân tích Dataset DrugBank Việt Nam Mở Rộng
Hiển thị thống kê và ví dụ về dataset đã bổ sung thuốc Việt Nam
"""

import json
from collections import Counter

def analyze_extended_dataset():
    print("🇻🇳 PHÂN TÍCH DATASET DRUGBANK VIỆT NAM MỞ RỘNG")
    print("=" * 70)
    
    # Đọc dataset mở rộng
    train_file = "drugbank_qa_vietnamese_extended_train.jsonl"
    test_file = "drugbank_qa_vietnamese_extended_test.jsonl" 
    analysis_file = "drugbank_qa_vietnamese_extended_analysis.json"
    
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    # Thống kê tổng quan
    print(f"📊 THỐNG KÊ TỔNG QUAN:")
    print(f"   • Dataset gốc (DrugBank): {analysis['original_dataset']['total_samples']:,} mẫu")
    print(f"   • Thuốc Việt Nam bổ sung: {analysis['vietnamese_addition']['total_samples']:,} mẫu")  
    print(f"   • Dataset mở rộng: {analysis['extended_dataset']['total_samples']:,} mẫu")
    print(f"   • Tăng thêm: {analysis['vietnamese_addition']['total_samples']}/{analysis['original_dataset']['total_samples']*100:.1f}%")
    
    # Phân tích cân bằng
    train_answers = Counter(item['output'] for item in train_data)
    test_answers = Counter(item['output'] for item in test_data)
    
    print(f"\n🎯 CÂN BẰNG DỮ LIỆU:")
    print(f"   Training Set:")
    for answer, count in train_answers.items():
        pct = count/len(train_data)*100
        print(f"      • {answer}: {count:,} mẫu ({pct:.1f}%)")
    
    print(f"   Test Set:")
    for answer, count in test_answers.items():
        pct = count/len(test_data)*100  
        print(f"      • {answer}: {count:,} mẫu ({pct:.1f}%)")
    
    # Phân loại các loại thuốc
    print(f"\n💊 PHÂN LOẠI THUỐC ĐÃ BỔ SUNG:")
    
    # Thuốc thông dụng tiếng Anh
    print("   Thuốc thông dụng (tên quốc tế):")
    for drug in analysis['vietnamese_drug_examples']:
        print(f"      • {drug}")
    
    # Thuốc Việt hóa  
    print("   Thuốc Việt hóa (tên dân gian):")
    for name in analysis['vietnamese_name_examples']:
        print(f"      • {name}")
    
    # Tìm ví dụ cụ thể trong dataset
    print(f"\n🌟 VÍ DỤ CỤ THỂ TRONG DATASET:")
    
    # Ví dụ thuốc thông dụng
    common_drugs = ['Paracetamol', 'Aspirin', 'Vitamin C', 'Amoxicillin']
    for drug in common_drugs[:3]:
        examples = [item for item in train_data if drug in item['input']]
        if examples:
            ex = examples[0]
            print(f"   📝 {drug}:")
            print(f"      Q: {ex['input']}")
            print(f"      A: {ex['output']}")
            print()
    
    # Ví dụ thuốc Việt hóa
    vietnamese_phrases = ['Thuốc giảm đau', 'Thuốc dạ dày', 'Kháng sinh']
    for phrase in vietnamese_phrases[:2]:
        examples = [item for item in train_data if phrase in item['input']]
        if examples:
            ex = examples[0]
            print(f"   📝 {phrase}:")
            print(f"      Q: {ex['input']}")
            print(f"      A: {ex['output']}")
            print()
    
    # Đánh giá độ phủ
    print(f"📈 ĐỘ PHỦ THUỐC VIỆT NAM:")
    
    # Đếm câu hỏi có chứa thuốc phổ biến
    common_drug_count = 0
    vietnamese_phrase_count = 0
    
    all_data = train_data + test_data
    
    for item in all_data:
        text = item['input'].lower()
        
        # Thuốc phổ biến
        common_drugs_lower = [d.lower() for d in ['paracetamol', 'aspirin', 'vitamin', 'amoxicillin', 'ibuprofen']]
        if any(drug in text for drug in common_drugs_lower):
            common_drug_count += 1
        
        # Cụm từ Việt hóa
        vietnamese_phrases_lower = ['thuốc giảm đau', 'thuốc dạ dày', 'kháng sinh', 'vitamin']
        if any(phrase in text for phrase in vietnamese_phrases_lower):
            vietnamese_phrase_count += 1
    
    total_samples = len(all_data)
    print(f"   • Câu hỏi chứa thuốc phổ biến: {common_drug_count:,}/{total_samples:,} ({common_drug_count/total_samples*100:.1f}%)")
    print(f"   • Câu hỏi chứa thuật ngữ Việt: {vietnamese_phrase_count:,}/{total_samples:,} ({vietnamese_phrase_count/total_samples*100:.1f}%)")
    
    # Gợi ý sử dụng
    print(f"\n🚀 KHUYẾN NGHỊ SỬ DỤNG:")
    print(f"   • Dataset này phù hợp để test với tên thuốc Việt Nam phổ biến")
    print(f"   • Bao gồm cả tên khoa học và tên thông dụng")
    print(f"   • Có thể xử lý câu hỏi về thuốc bằng tiếng Việt tự nhiên")
    print(f"   • Sẵn sàng cho training với Gemma/Qwen trên dữ liệu Việt Nam")
    
    print(f"\n📁 FILES ĐƯỢC TẠO:")
    print(f"   • {train_file} ({len(train_data):,} mẫu training)")
    print(f"   • {test_file} ({len(test_data):,} mẫu test)")
    print(f"   • {analysis_file} (thống kê chi tiết)")

if __name__ == "__main__":
    analyze_extended_dataset()