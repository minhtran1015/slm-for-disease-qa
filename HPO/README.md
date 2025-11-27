# HPO Vietnamese Symptom Relationship Dataset

## Overview

Vietnamese symptom relationship dataset for training SLMs on medical ontology understanding. Generated from Human Phenotype Ontology (HPO) using graph traversal and GPU-accelerated translation to create yes/no questions about symptom hierarchies.

## 📊 Available Versions

| Version | Samples | Format | Recommended |
|---------|---------|--------|-------------|
| **Bilingual** | 20,000 (18k train + 2k test) | Vietnamese (English) | ✅ Yes |
| **Basic** | 19,998 (18k train + 2k test) | Vietnamese only | Legacy |

**Key Features**:

- **20,000 samples** total (18,000 training + 2,000 test)
- **Perfect balance**: 10,000 True + 10,000 False (50/50)
- **Bilingual format**: Vietnamese (English) preserves original medical terms
- **8 instruction templates** + **8 question templates** for variety
- **GPU-accelerated**: Modal A10G + NLLB-200 translation (~2.5 min processing)

## Data Structure

**Bilingual Vietnamese Symptom QA Format** (Recommended):

```json
{
  "instruction": "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau.",
  "input": "Suy tim (Heart failure) có phải là một dạng của Rối loạn tim mạch (Cardiovascular abnormality) không?",
  "output": "Đúng",
  "question_type": "true_relationship",
  "child_en": "Heart failure",
  "child_vi": "Suy tim",
  "parent_en": "Cardiovascular abnormality",
  "parent_vi": "Rối loạn tim mạch"
}
```

## Template Variety

### Instruction Templates (8 variants)

- "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai."
- "Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa."
- "Xác định tính đúng sai của nhận định sau về triệu chứng y học."
- "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau."
- "Dựa vào phân loại triệu chứng y học, hãy trả lời Đúng hoặc Sai."
- "Với kiến thức về bệnh học, hãy xác nhận câu sau Đúng hay Sai."
- "Câu hỏi về mối quan hệ triệu chứng - Trả lời Đúng hoặc Sai."
- "Theo hệ thống phân loại y khoa, câu sau Đúng hay Sai?"

### Question Templates (8 variants)

- "... có phải là một dạng của ... không?"
- "Trong y học, ... thuộc nhóm ... đúng không?"
- "Xác nhận: ... là biểu hiện liên quan đến ...?"
- "... có thuộc loại ... không?"
- "Triệu chứng ... có nằm trong nhóm ... không?"
- "... có liên quan đến ... không?"
- "Biểu hiện ... có phải là một phần của ... không?"
- "... được xếp vào loại ... phải không?"

## 🚀 Quick Start

### Using the Dataset

```python
import json

def load_jsonl(filepath):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# Load bilingual dataset (recommended)
train_data = load_jsonl('hpo_vietnamese_bilingual_train.jsonl')
test_data = load_jsonl('hpo_vietnamese_bilingual_test.jsonl')

print(f"Train: {len(train_data)}, Test: {len(test_data)}")
# Output: Train: 18000, Test: 2000
```

### Regenerating the Dataset

```bash
# Install Modal
pip install modal
modal setup

# Upload HPO source data
modal volume put medical-data hp.json hp.json

# Run translation pipeline
modal run --detach convert_hpo_bilingual_modal.py

# Download results
modal volume get medical-data hpo_vietnamese_bilingual_train.jsonl ./
modal volume get medical-data hpo_vietnamese_bilingual_test.jsonl ./
```

## Translation Pipeline

### Configuration

```python
# convert_hpo_bilingual_modal.py settings
TARGET_PAIRS = 10000      # 10k true + 10k false = 20k total
BATCH_SIZE = 128          # Terms per GPU batch
MODEL = "facebook/nllb-200-distilled-600M"
GPU = "A10G"              # Modal GPU type
```

### Processing Time

- **Translation**: ~2.5 minutes for 13,530 unique terms
- **Cost**: ~$0.05 (Modal A10G pricing)

## File Structure

```
HPO/
├── hp.json                                    # Source HPO ontology (21MB)
├── convert_hpo_bilingual_modal.py             # GPU translation pipeline
├── convert_hpo_to_vietnamese.py               # Basic version (legacy)
├── hpo_vietnamese_bilingual_train.jsonl       # Bilingual train (18,000)
├── hpo_vietnamese_bilingual_test.jsonl        # Bilingual test (2,000)
├── hpo_vietnamese_bilingual_stats.json        # Statistics
├── hpo_vietnamese_symptoms_train.jsonl        # Basic train (legacy)
├── hpo_vietnamese_symptoms_test.jsonl         # Basic test (legacy)
└── README.md                                  # This documentation
```

## Dataset Statistics

### Bilingual Version (Recommended)

- **Total Samples**: 20,000
- **Training Set**: 18,000 (90%)
- **Test Set**: 2,000 (10%)
- **Balance**: 10,000 True : 10,000 False (50:50)
- **Unique Terms Translated**: 13,530
- **Translation Model**: NLLB-200-distilled-600M

## Training Configuration

**Recommended for Vietnamese SLMs**:

```python
LEARNING_RATES = {
    "gemma-1b": 2e-5,
    "qwen-0.5b": 3e-5,
    "vistral-7b": 1e-5
}

BATCH_SIZES = {
    "gemma-1b": 16,
    "qwen-0.5b": 32,
    "vistral-7b": 8
}

MAX_LENGTH = 256
EPOCHS = 3-5
```

## Key Advantages

1. **Bilingual Format**: Vietnamese (English) preserves original medical terms
2. **Template Diversity**: 8×8 = 64 unique instruction-question combinations
3. **Perfect Balance**: 50% Đúng / 50% Sai prevents bias
4. **GPU-Accelerated**: Fast regeneration via Modal cloud
5. **Ontology-Based**: Leverages HPO's structured medical knowledge

## Common Use Cases

- Vietnamese medical diagnosis assistants
- Symptom checking applications
- Medical education tools
- Clinical decision support systems
- Automated medical coding

## Acknowledgments

- **HPO (Human Phenotype Ontology)**: Source ontology data
- **NLLB-200**: Facebook's translation model
- **Modal**: GPU cloud infrastructure