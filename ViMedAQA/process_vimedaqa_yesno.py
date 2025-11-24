import json
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ================= CẤU HÌNH =================
OUTPUT_FILE = 'vimedaqa_yesno_50k.jsonl'
SUBSETS = ['disease', 'drug', 'body-part'] # Các phần của ViMedAQA
SEED = 42
random.seed(SEED)

def format_yes_no_prompt(question, proposed_answer):
    """
    Tạo template câu hỏi Yes/No tự nhiên nhất cho tiếng Việt.
    Thay vì nối chuỗi vụng về, ta dùng format kiểm chứng thông tin.
    """
    # Template: Đưa ra ngữ cảnh và hỏi model xác nhận
    return {
        "context": f"Câu hỏi: {question}\nThông tin tham khảo: {proposed_answer}",
        "question": "Dựa vào thông tin tham khảo, câu trả lời trên là ĐÚNG hay SAI so với câu hỏi?",
    }

def process_vimedaqa():
    all_data = []
    
    print("🚀 Đang tải dữ liệu từ Hugging Face...")
    
    # 1. Tải và gộp dữ liệu
    for subset in SUBSETS:
        try:
            # Tải tập train
            ds = load_dataset("tmnam20/ViMedAQA", subset, split='train')
            df = ds.to_pandas()
            
            # Chỉ lấy các cột cần thiết và lọc bỏ dữ liệu trống
            df = df[['question', 'answer']].dropna()
            
            # Lưu lại danh sách các câu trả lời của nhóm này để làm mẫu sai (Negative Sampling)
            # Việc lấy mẫu sai trong cùng 1 nhóm (VD: Thuốc với Thuốc) sẽ khó hơn là lấy khác nhóm, giúp model học tốt hơn.
            category_answers = df['answer'].tolist()
            
            print(f"   - Đang xử lý nhóm '{subset}': {len(df)} dòng gốc...")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   Processing {subset}"):
                q_text = row['question'].strip()
                a_true = row['answer'].strip()
                
                if not q_text or not a_true:
                    continue

                # --- A. TẠO MẪU ĐÚNG (POSITIVE SAMPLE) ---
                prompt_true = format_yes_no_prompt(q_text, a_true)
                all_data.append({
                    "instruction": prompt_true['question'],
                    "input": prompt_true['context'],
                    "output": "Đúng"
                })

                # --- B. TẠO MẪU SAI (NEGATIVE SAMPLE) ---
                # Random một câu trả lời khác trong cùng danh sách
                attempts = 0
                max_attempts = 10  # Prevent infinite loop
                while attempts < max_attempts:
                    a_false = random.choice(category_answers).strip()
                    # Đảm bảo câu trả lời giả không trùng với câu trả lời thật
                    if a_false != a_true and len(a_false) > 5: 
                        break
                    attempts += 1
                
                if attempts < max_attempts:  # Only add if we found a valid false answer
                    prompt_false = format_yes_no_prompt(q_text, a_false)
                    all_data.append({
                        "instruction": prompt_false['question'],
                        "input": prompt_false['context'],
                        "output": "Sai"
                    })

        except Exception as e:
            print(f"⚠️ Lỗi khi tải nhóm {subset}: {e}")

    # 2. Xáo trộn dữ liệu (Shuffle)
    random.shuffle(all_data)
    
    # 3. Lưu ra file
    print(f"\n💾 Đang lưu {len(all_data)} mẫu dữ liệu vào '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print("✅ Hoàn tất! File đã sẵn sàng để train.")
    print(f"   Tổng số lượng mẫu: {len(all_data)}")
    
    # Calculate statistics
    positive_samples = sum(1 for entry in all_data if entry['output'] == 'Đúng')
    negative_samples = sum(1 for entry in all_data if entry['output'] == 'Sai')
    
    print(f"   Mẫu ĐÚNG: {positive_samples}")
    print(f"   Mẫu SAI: {negative_samples}")
    print(f"   Tỷ lệ cân bằng: {positive_samples/len(all_data):.1%} / {negative_samples/len(all_data):.1%}")
    
    print("   Ví dụ mẫu đầu tiên:")
    print(json.dumps(all_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    process_vimedaqa()