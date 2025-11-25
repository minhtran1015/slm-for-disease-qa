# Dataset DrugBank Việt Nam Mở Rộng 🇻🇳

## Tổng Quan

Dataset này được thiết kế đặc biệt cho việc training các Small Language Models (SLMs) như Gemma 1B và Qwen 0.5B trên tác vụ nhận dạng thuốc bằng tiếng Việt. Dataset bao gồm cả tên thuốc khoa học từ DrugBank và các tên thuốc Việt hóa phổ biến.

## 📊 Thống Kê Dataset

- **Tổng cộng: 19,754 mẫu**
  - Training: 17,780 mẫu (90%)
  - Test: 1,974 mẫu (10%)

- **Nguồn dữ liệu:**
  - DrugBank khoa học: 19,528 mẫu
  - Thuốc Việt Nam bổ sung: 226 mẫu

- **Cân bằng hoàn hảo:** ~50% Đúng / ~50% Sai

## 🏥 Các Loại Thuốc Được Bao Gồm

### Thuốc Thông Dụng (Tên Quốc Tế)
- **Giảm đau, hạ sốt:** Paracetamol, Aspirin, Ibuprofen, Diclofenac
- **Kháng sinh:** Amoxicillin, Augmentin, Ciprofloxacin, Azithromycin  
- **Dạ dày:** Omeprazole, Esomeprazole, Domperidone
- **Tim mạch:** Amlodipine, Losartan, Atorvastatin
- **Đái tháo đường:** Metformin, Glibenclamide, Insulin
- **Vitamin:** Vitamin C, Vitamin D3, Vitamin B Complex

### Thuốc Việt Hóa (Tên Dân Gian)
- "Thuốc giảm đau Paracetamol"
- "Thuốc dạ dày Omeprazole" 
- "Kháng sinh Amoxicillin"
- "Thuốc huyết áp Amlodipine"
- "Thuốc chống dị ứng"
- "Vitamin tăng cường miễn dịch"
- "Thuốc an thần"
- "Men vi sinh"

## 📝 Format Dữ Liệu

```json
{
  "instruction": "Dựa vào kiến thức về thuốc và y học, hãy trả lời Đúng hoặc Sai.",
  "input": "Paracetamol có phải là một loại thuốc giảm đau không?",
  "output": "Đúng"
}
```

### Template Câu Hỏi Tiếng Việt

**Positive (Đúng):**
- "{name} có phải là một loại thuốc không?"
- "{name} được sử dụng để điều trị bệnh phải không?"
- "Có thể mua {name} tại hiệu thuốc không?"
- "Bác sĩ có thể kê đơn {name} không?"

**Negative (Sai):**
- "{name} có phải là một loại bệnh không?"
- "{name} là triệu chứng của bệnh gì đó phải không?"
- "{name} có phải là tên một cơ quan trong cơ thể không?"
- "{name} là một loại vi khuẩn gây bệnh phải không?"

## 🚀 Cách Sử Dụng

### 1. Training Cơ Bản

```python
# Load dataset
from datasets import Dataset
import json

def load_vietnamese_drug_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load training data
train_data = load_vietnamese_drug_dataset('drugbank_qa_vietnamese_extended_train.jsonl')
test_data = load_vietnamese_drug_dataset('drugbank_qa_vietnamese_extended_test.jsonl')

# Convert to HF format
texts = [f"{item['instruction']} {item['input']}" for item in train_data]
labels = [1 if item['output'] == 'Đúng' else 0 for item in train_data]
```

### 2. Training với Script Có Sẵn

```bash
# Train với Gemma 1B
python train_vietnamese_extended.py

# Hoặc sửa MODEL_NAME trong script thành:
# "Qwen/Qwen2.5-0.5B" cho Qwen 0.5B
```

### 3. Inference với Model Đã Train

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./vietnamese-drugbank-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_vietnamese_drug(instruction, question):
    text = f"{instruction} {question}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return "Đúng" if predicted_class == 1 else "Sai"

# Test
result = predict_vietnamese_drug(
    "Dựa vào kiến thức về thuốc, trả lời Đúng hoặc Sai.",
    "Paracetamol có phải là thuốc giảm đau không?"
)
print(result)  # Output: "Đúng"
```

## 📁 Files Trong Dataset

- `drugbank_qa_vietnamese_extended_train.jsonl` - Dữ liệu training (17,780 mẫu)
- `drugbank_qa_vietnamese_extended_test.jsonl` - Dữ liệu test (1,974 mẫu)
- `drugbank_qa_vietnamese_extended_analysis.json` - Thống kê chi tiết
- `train_vietnamese_extended.py` - Script training hoàn chỉnh
- `analyze_extended_dataset.py` - Script phân tích dataset

## 🎯 Ưu Điểm Của Dataset

### 1. Phù Hợp với Test Data Việt Nam
- Bao gồm tên thuốc phổ biến trong dân gian
- Xử lý được cả tên khoa học và tên Việt hóa
- Phản ánh cách người Việt gọi tên thuốc trong thực tế

### 2. Cân Bằng Hoàn Hảo
- 50/50 positive/negative samples
- Tránh bias trong quá trình học
- Đảm bảo model học cả hai class đều tốt

### 3. Đa Dạng Template
- 10 template câu hỏi positive
- 10 template câu hỏi negative  
- 5 template instruction khác nhau
- Tăng khả năng generalization

### 4. Chất Lượng Cao
- Dữ liệu từ DrugBank (cơ sở dữ liệu thuốc uy tín)
- Thuốc Việt Nam được chọn lọc kỹ càng
- Câu hỏi tự nhiên, phản ánh cách hỏi thực tế

## 🔧 Tối Ưu Hóa Training

### Cho Gemma 1B:
- Learning rate: 2e-5 đến 5e-5
- Batch size: 16-32
- Epochs: 3-5
- Max sequence length: 512

### Cho Qwen 0.5B:
- Learning rate: 3e-5 đến 1e-4
- Batch size: 32-64  
- Epochs: 5-8
- Max sequence length: 512

## 📈 Kết Quả Mong Đợi

Sau khi train trên dataset này, model sẽ có khả năng:
- Nhận dạng chính xác các tên thuốc tiếng Việt phổ biến
- Phân biệt thuốc và các thực thể khác (bệnh, triệu chứng, cơ quan...)
- Xử lý cả tên khoa học và tên dân gian
- Đạt accuracy > 90% trên test data Việt Nam

## 🌟 Use Cases

- **Hệ thống tư vấn thuốc tự động**
- **Chatbot y tế tiếng Việt**
- **Công cụ kiểm tra thông tin thuốc**
- **Hỗ trợ dược sĩ và bác sĩ**
- **Ứng dụng tra cứu thuốc cho người dân**

---

**Dataset này sẵn sàng để đối phó với test data có tên thuốc Việt Nam phổ biến!** 🎯🇻🇳