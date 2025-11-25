#!/usr/bin/env python3
"""
Training script cho Dataset DrugBank Việt Nam Mở Rộng
Bao gồm cả thuốc khoa học và thuốc Việt hóa phổ biến
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np

class VietnameseDrugDataset(Dataset):
    """PyTorch Dataset cho dữ liệu thuốc Việt Nam."""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Kết hợp instruction và input
        text = f"{item['instruction']} {item['input']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Chuyển Đúng/Sai thành binary labels
        label = 1 if item['output'] == 'Đúng' else 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_vietnamese_dataset(jsonl_file, tokenizer, max_length=512):
    """Load dữ liệu JSONL và chuyển sang Hugging Face Dataset format."""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Chuẩn bị texts và labels
    texts = [f"{item['instruction']} {item['input']}" for item in data]
    labels = [1 if item['output'] == 'Đúng' else 0 for item in data]
    
    # Tokenize tất cả texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Tạo HF Dataset
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }
    
    return HFDataset.from_dict(dataset_dict)

def compute_metrics(eval_pred):
    """Tính metrics cho evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    """Hàm training chính."""
    
    # Cấu hình
    MODEL_NAME = "google/gemma-2b"  # Đổi thành "Qwen/Qwen2.5-0.5B" cho Qwen
    TRAIN_FILE = "drugbank_qa_vietnamese_extended_train.jsonl"
    TEST_FILE = "drugbank_qa_vietnamese_extended_test.jsonl"  
    OUTPUT_DIR = "./vietnamese-drugbank-model"
    MAX_LENGTH = 512
    
    print(f"🇻🇳 Bắt đầu training với model: {MODEL_NAME}")
    print(f"📊 Sử dụng dataset mở rộng (bao gồm thuốc Việt Nam)")
    
    # Load tokenizer và model
    print("📥 Đang load tokenizer và model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Thêm padding token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Chuẩn bị datasets
    print("📊 Đang chuẩn bị datasets...")
    train_dataset = load_vietnamese_dataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    test_dataset = load_vietnamese_dataset(TEST_FILE, tokenizer, MAX_LENGTH)
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    
    # Phân tích dataset
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    dung_count = sum(1 for item in train_data if item['output'] == 'Đúng')
    sai_count = len(train_data) - dung_count
    
    print(f"   Đúng: {dung_count:,} ({dung_count/len(train_data)*100:.1f}%)")
    print(f"   Sai: {sai_count:,} ({sai_count/len(train_data)*100:.1f}%)")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Tắt wandb/tensorboard logging
        save_total_limit=2,
        dataloader_pin_memory=False,
    )
    
    # Khởi tạo trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("🏋️ Bắt đầu training...")
    trainer.train()
    
    # Evaluation cuối cùng
    print("📊 Chạy evaluation cuối cùng...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ Training hoàn thành!")
    print(f"📈 Kết quả cuối cùng:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '').title()
            print(f"   {metric_name}: {value:.4f}")
    
    # Lưu model
    print(f"💾 Đang lưu model vào: {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n🎉 Model sẵn sàng cho inference!")
    print(f"Load bằng: AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")

def vietnamese_inference_example():
    """Ví dụ cách sử dụng model đã train cho inference với thuốc Việt Nam."""
    
    example_code = '''
# Load trained model cho thuốc Việt Nam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./vietnamese-drugbank-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Hàm predict cho thuốc Việt Nam
def predict_vietnamese_drug(instruction, question):
    text = f"{instruction} {question}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return "Đúng" if predicted_class == 1 else "Sai", probabilities[0][predicted_class].item()

# Test với thuốc Việt Nam
examples = [
    ("Dựa vào kiến thức về thuốc, hãy trả lời Đúng hoặc Sai.", 
     "Paracetamol có phải là thuốc giảm đau không?"),
    ("Xác định xem đây có phải là thuốc. Trả lời Đúng hoặc Sai.", 
     "Thuốc dạ dày Omeprazole có phải là loại virus không?"),
    ("Sử dụng hiểu biết về dược phẩm, trả lời Đúng hoặc Sai.",
     "Kháng sinh Amoxicillin có được bán tại nhà thuốc không?"),
    ("Đánh giá xem đây có phải thuốc. Trả lời Đúng hoặc Sai.",
     "Vitamin C có phải là một bệnh truyền nhiễm không?")
]

print("🇻🇳 Test với thuốc Việt Nam:")
for instruction, question in examples:
    answer, confidence = predict_vietnamese_drug(instruction, question)
    print(f"Q: {question}")
    print(f"A: {answer} (độ tin cậy: {confidence:.3f})")
    print()
'''
    print("\n🔮 Ví dụ Inference với Thuốc Việt Nam:")
    print("=" * 60)
    print(example_code)

if __name__ == "__main__":
    # Chạy training
    main()
    
    # Hiển thị ví dụ inference
    vietnamese_inference_example()