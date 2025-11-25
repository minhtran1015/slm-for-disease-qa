# DrugBank Vietnamese QA Dataset 🇻🇳

Vietnamese drug identification dataset for training Small Language Models on medical Yes/No questions.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate base dataset from DrugBank
python generate_vietnamese_qa_dataset.py --samples 20000

# Add Vietnamese common drug names
python prepare_vietnamese_drugs.py

# Train model
python train_vietnamese_qa.py

# Analyze dataset
python analyze_vietnamese_dataset.py
```

## 📊 Dataset

- **19,754 samples** (17,780 train + 1,974 test)
- **Perfect balance**: 50% Đúng / 50% Sai
- **Bilingual**: Scientific + Vietnamese drug names
- **Format**: Instruction-following JSONL

## 📁 Key Files

- `drugbank_qa_vietnamese_extended_train.jsonl` - Training data
- `drugbank_qa_vietnamese_extended_test.jsonl` - Test data
- `train_vietnamese_qa.py` - Complete training script
- `README_VIETNAMESE_EXTENDED.md` - **Detailed documentation**

## 💊 Drug Coverage

- **International drugs**: Paracetamol, Aspirin, Amoxicillin...
- **Vietnamese names**: "Thuốc giảm đau", "Thuốc dạ dày"...
- **65+ common drugs** + **50+ Vietnamese phrases**

## 🎯 Compatible Models

- Gemma 1B/2B
- Qwen 0.5B/1.8B
- Vistral 7B
- Any Vietnamese-capable SLM

---
📖 **See [README_VIETNAMESE_EXTENDED.md](README_VIETNAMESE_EXTENDED.md) for complete documentation**