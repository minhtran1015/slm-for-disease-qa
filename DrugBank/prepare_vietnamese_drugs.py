#!/usr/bin/env python3
"""
Vietnamese Drug Names Supplement
Bổ sung các tên thuốc dân gian và Việt hóa phổ biến vào dataset DrugBank
"""

import json
import random

# Danh sách các thuốc thông dụng bằng tiếng Việt
VIETNAMESE_DRUGS = [
    # Thuốc giảm đau, hạ sốt phổ biến
    "Paracetamol", "Aspirin", "Ibuprofen", "Diclofenac", 
    "Ketoprofen", "Naproxen", "Celecoxib",
    
    # Thuốc kháng sinh phổ biến
    "Amoxicillin", "Augmentin", "Ciprofloxacin", "Azithromycin",
    "Cephalexin", "Doxycycline", "Metronidazole", "Clarithromycin",
    
    # Thuốc dạ dày
    "Omeprazole", "Esomeprazole", "Lansoprazole", "Ranitidine",
    "Domperidone", "Simethicone", "Sucralfate",
    
    # Thuốc tim mạch
    "Amlodipine", "Losartan", "Enalapril", "Metoprolol",
    "Atorvastatin", "Simvastatin", "Clopidogrel", "Warfarin",
    
    # Thuốc đái tháo đường
    "Metformin", "Glibenclamide", "Gliclazide", "Insulin",
    "Pioglitazone", "Sitagliptin",
    
    # Thuốc hen suyễn, dị ứng
    "Salbutamol", "Prednisolone", "Cetirizine", "Loratadine",
    "Montelukast", "Budesonide",
    
    # Vitamin và khoáng chất
    "Vitamin C", "Vitamin D3", "Vitamin B1", "Vitamin B6",
    "Vitamin B12", "Acid Folic", "Canxi", "Sắt", "Kẽm",
    
    # Thuốc thần kinh
    "Diazepam", "Alprazolam", "Sertraline", "Fluoxetine",
    "Risperidone", "Olanzapine", "Gabapentin",
    
    # Thuốc da liễu
    "Hydrocortisone", "Betamethasone", "Clotrimazole", "Ketoconazole",
    
    # Các thuốc khác thường dùng
    "Chlorpheniramine", "Dextromethorphan", "Loperamide",
    "Bisacodyl", "Paracetamol + Caffeine"
]

# Tên thuốc Việt hóa phổ biến
VIETNAMESE_NAMES = [
    # Thuốc giảm đau
    ("Thuốc giảm đau Paracetamol", "Paracetamol"),
    ("Thuốc hạ sốt cho trẻ em", "Paracetamol dạng siro"),
    ("Aspirin bảo vệ tim", "Aspirin"),
    ("Thuốc chống viêm Ibuprofen", "Ibuprofen"),
    
    # Thuốc kháng sinh
    ("Kháng sinh Amoxicillin", "Amoxicillin"),
    ("Thuốc kháng viêm Augmentin", "Amoxicillin + Clavulanate"),
    ("Kháng sinh đường uống Azithromycin", "Azithromycin"),
    ("Thuốc trị nhiễm khuẩn Ciprofloxacin", "Ciprofloxacin"),
    
    # Thuốc dạ dày
    ("Thuốc dạ dày Omeprazole", "Omeprazole"),
    ("Thuốc chống acid dạ dày", "Omeprazole"),
    ("Thuốc trị đầy hơi", "Simethicone"),
    ("Thuốc chống nôn", "Domperidone"),
    
    # Thuốc tim mạch
    ("Thuốc huyết áp Amlodipine", "Amlodipine"),
    ("Thuốc điều trị cao huyết áp", "Losartan"),
    ("Thuốc chống đông máu", "Warfarin"),
    ("Thuốc giảm cholesterol", "Atorvastatin"),
    
    # Thuốc đái tháo đường
    ("Thuốc đái tháo đường Metformin", "Metformin"),
    ("Thuốc hạ đường huyết", "Glibenclamide"),
    ("Insulin tiêm", "Insulin"),
    
    # Thuốc ho, cảm
    ("Thuốc ho Khan", "Dextromethorphan"),
    ("Thuốc giãn phế quản", "Salbutamol"),
    ("Thuốc xịt mũi", "Budesonide"),
    ("Thuốc cảm cúm", "Paracetamol + Pseudoephedrine"),
    
    # Thuốc dị ứng
    ("Thuốc chống dị ứng", "Cetirizine"),
    ("Thuốc trị mày đay", "Loratadine"),
    ("Thuốc corticoid", "Prednisolone"),
    
    # Vitamin
    ("Vitamin tăng cường miễn dịch", "Vitamin C"),
    ("Vitamin D cho xương", "Vitamin D3"),
    ("Vitamin nhóm B", "Vitamin B Complex"),
    ("Canxi bổ sung", "Calcium Carbonate"),
    ("Sắt bổ máu", "Iron Sulfate"),
    
    # Thuốc thần kinh
    ("Thuốc an thần", "Diazepam"),
    ("Thuốc chống trầm cảm", "Sertraline"),
    ("Thuốc ngủ", "Zolpidem"),
    
    # Thuốc tiêu hóa
    ("Thuốc tiêu hóa", "Pancreatin"),
    ("Thuốc chống tiêu chảy", "Loperamide"),
    ("Thuốc nhuận tràng", "Bisacodyl"),
    ("Men vi sinh", "Lactobacillus"),
    
    # Thuốc da liễu
    ("Thuốc bôi da", "Hydrocortisone"),
    ("Thuốc trị nám", "Tretinoin"),
    ("Thuốc trị nấm", "Clotrimazole"),
    
    # Thuốc phụ khoa
    ("Thuốc tránh thai", "Levonorgestrel + Ethinylestradiol"),
    ("Thuốc nội tiết tố", "Estradiol"),
    
    # Thuốc mắt tai mũi họng
    ("Thuốc nhỏ mắt", "Chloramphenicol eye drops"),
    ("Thuốc xịt họng", "Benzydamine"),
    ("Thuốc nhỏ tai", "Ciprofloxacin ear drops"),
]

def generate_vietnamese_drug_samples():
    """Tạo các mẫu QA cho thuốc Việt Nam"""
    
    # Template câu hỏi tiếng Việt
    POSITIVE_TEMPLATES_VN = [
        "{name} có phải là một loại thuốc không?",
        "{name} được sử dụng để điều trị bệnh phải không?",
        "Có thể mua {name} tại hiệu thuốc không?",
        "{name} là một loại dược phẩm đúng không?",
        "Bác sĩ có thể kê đơn {name} không?",
        "{name} có tác dụng chữa bệnh không?",
        "{name} thuộc nhóm thuốc điều trị phải không?",
        "Người bệnh có thể sử dụng {name} không?",
        "{name} có được bán trong nhà thuốc không?",
        "{name} là thuốc được cấp phép lưu hành không?"
    ]
    
    NEGATIVE_TEMPLATES_VN = [
        "{name} có phải là một loại bệnh không?",
        "{name} là triệu chứng của bệnh gì đó phải không?",
        "{name} có phải là tên một cơ quan trong cơ thể không?",
        "{name} là một loại vi khuẩn gây bệnh phải không?",
        "{name} có phải là phương pháp phẫu thuật không?",
        "{name} là tên một xét nghiệm y tế phải không?",
        "{name} có phải là thiết bị y tế không?",
        "{name} là một hội chứng bệnh lý phải không?",
        "{name} có phải là virus gây bệnh không?",
        "{name} là tên một loại ung thư phải không?"
    ]
    
    INSTRUCTION_TEMPLATES_VN = [
        "Dựa vào kiến thức về thuốc và y học, hãy trả lời Đúng hoặc Sai.",
        "Xác định xem đây có phải là thuốc hay không. Trả lời Đúng hoặc Sai.",
        "Sử dụng hiểu biết về dược phẩm, hãy trả lời Đúng hoặc Sai.",
        "Đánh giá xem đây có phải là một loại thuốc. Trả lời Đúng hoặc Sai.",
        "Dựa trên kiến thức y dược, hãy trả lời Đúng hoặc Sai."
    ]
    
    samples = []
    all_drugs = VIETNAMESE_DRUGS + [vn_name for vn_name, _ in VIETNAMESE_NAMES]
    
    for drug in all_drugs:
        # Tạo câu hỏi positive (Đúng)
        positive_template = random.choice(POSITIVE_TEMPLATES_VN)
        positive_question = positive_template.format(name=drug)
        instruction = random.choice(INSTRUCTION_TEMPLATES_VN)
        
        samples.append({
            "instruction": instruction,
            "input": positive_question,
            "output": "Đúng",
            "drug_name": drug,
            "sample_type": "positive_vietnamese"
        })
        
        # Tạo câu hỏi negative (Sai)  
        negative_template = random.choice(NEGATIVE_TEMPLATES_VN)
        negative_question = negative_template.format(name=drug)
        instruction = random.choice(INSTRUCTION_TEMPLATES_VN)
        
        samples.append({
            "instruction": instruction,
            "input": negative_question,
            "output": "Sai",
            "drug_name": drug,
            "sample_type": "negative_vietnamese"
        })
    
    return samples

def add_vietnamese_drugs_to_dataset():
    """Bổ sung thuốc Việt Nam vào dataset hiện tại"""
    
    print("🇻🇳 Đang bổ sung các thuốc Việt Nam phổ biến...")
    
    # Tạo các mẫu mới
    vietnamese_samples = generate_vietnamese_drug_samples()
    
    # Shuffle các mẫu mới
    random.shuffle(vietnamese_samples)
    
    print(f"   • Đã tạo {len(vietnamese_samples)} mẫu thuốc Việt Nam")
    print(f"   • Positive: {len([s for s in vietnamese_samples if s['output'] == 'Đúng'])}")
    print(f"   • Negative: {len([s for s in vietnamese_samples if s['output'] == 'Sai'])}")
    
    # Đọc dataset hiện tại
    current_train_file = "drugbank_qa_vietnamese_20k.jsonl"
    current_test_file = "drugbank_qa_vietnamese_20k_test.jsonl"
    
    current_train_data = []
    with open(current_train_file, 'r', encoding='utf-8') as f:
        for line in f:
            current_train_data.append(json.loads(line))
    
    current_test_data = []
    with open(current_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            current_test_data.append(json.loads(line))
    
    print(f"\n📊 Dataset hiện tại:")
    print(f"   • Training: {len(current_train_data):,} mẫu")
    print(f"   • Test: {len(current_test_data):,} mẫu")
    
    # Chia mẫu Vietnamese thành train/test (90/10)
    test_size = int(len(vietnamese_samples) * 0.1)
    vietnamese_test = vietnamese_samples[:test_size]
    vietnamese_train = vietnamese_samples[test_size:]
    
    # Kết hợp với dataset hiện tại
    extended_train_data = current_train_data + vietnamese_train
    extended_test_data = current_test_data + vietnamese_test
    
    # Shuffle lại toàn bộ
    random.shuffle(extended_train_data)
    random.shuffle(extended_test_data)
    
    print(f"\n📊 Dataset mở rộng:")
    print(f"   • Training: {len(extended_train_data):,} mẫu (+{len(vietnamese_train)})")
    print(f"   • Test: {len(extended_test_data):,} mẫu (+{len(vietnamese_test)})")
    print(f"   • Total: {len(extended_train_data) + len(extended_test_data):,} mẫu")
    
    # Lưu dataset mở rộng
    extended_train_file = "drugbank_qa_vietnamese_extended_train.jsonl"
    extended_test_file = "drugbank_qa_vietnamese_extended_test.jsonl"
    
    print(f"\n💾 Đang lưu dataset mở rộng...")
    
    with open(extended_train_file, 'w', encoding='utf-8') as f:
        for sample in extended_train_data:
            # Loại bỏ metadata để training
            clean_sample = {
                "instruction": sample["instruction"],
                "input": sample["input"], 
                "output": sample["output"]
            }
            json.dump(clean_sample, f, ensure_ascii=False)
            f.write('\n')
    
    with open(extended_test_file, 'w', encoding='utf-8') as f:
        for sample in extended_test_data:
            clean_sample = {
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": sample["output"]
            }
            json.dump(clean_sample, f, ensure_ascii=False)
            f.write('\n')
    
    # Tạo phân tích dataset mở rộng
    analysis = {
        "original_dataset": {
            "train_samples": len(current_train_data),
            "test_samples": len(current_test_data),
            "total_samples": len(current_train_data) + len(current_test_data)
        },
        "vietnamese_addition": {
            "train_samples": len(vietnamese_train),
            "test_samples": len(vietnamese_test),
            "total_samples": len(vietnamese_samples),
            "drugs_added": len(VIETNAMESE_DRUGS) + len(VIETNAMESE_NAMES)
        },
        "extended_dataset": {
            "train_samples": len(extended_train_data),
            "test_samples": len(extended_test_data), 
            "total_samples": len(extended_train_data) + len(extended_test_data)
        },
        "vietnamese_drug_examples": VIETNAMESE_DRUGS[:10],
        "vietnamese_name_examples": [name for name, _ in VIETNAMESE_NAMES[:10]],
        "sample_examples": {
            "vietnamese_positive": next((s for s in vietnamese_samples if s['output'] == 'Đúng'), None),
            "vietnamese_negative": next((s for s in vietnamese_samples if s['output'] == 'Sai'), None)
        }
    }
    
    analysis_file = "drugbank_qa_vietnamese_extended_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ {extended_train_file}")
    print(f"   ✅ {extended_test_file}")
    print(f"   ✅ {analysis_file}")
    
    # In ví dụ
    print(f"\n🌟 Ví dụ thuốc Việt Nam được bổ sung:")
    vn_positive = next((s for s in vietnamese_samples if s['output'] == 'Đúng'), None)
    print(f"   📝 POSITIVE:")
    print(f"      Instruction: {vn_positive['instruction']}")
    print(f"      Question: {vn_positive['input']}")
    print(f"      Answer: {vn_positive['output']}")
    
    vn_negative = next((s for s in vietnamese_samples if s['output'] == 'Sai'), None)
    print(f"\n   📝 NEGATIVE:")
    print(f"      Instruction: {vn_negative['instruction']}")
    print(f"      Question: {vn_negative['input']}")
    print(f"      Answer: {vn_negative['output']}")
    
    print(f"\n✅ Hoàn thành! Dataset giờ đã bao gồm:")
    print(f"   • Tên thuốc khoa học (DrugBank)")
    print(f"   • Tên thuốc thông dụng tiếng Việt")  
    print(f"   • Tên thuốc dân gian Việt hóa")
    print(f"   • Tổng cộng: {len(extended_train_data) + len(extended_test_data):,} mẫu")

if __name__ == "__main__":
    random.seed(42)  # Để có kết quả nhất quán
    add_vietnamese_drugs_to_dataset()