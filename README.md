# SLM for Disease QA

**Vietnamese Medical Question-Answering Dataset for Small Language Models**

A comprehensive research project for training Small Language Models (SLMs) on medical question-answering tasks in Vietnamese. This project creates high-quality, balanced datasets from international medical ontologies and databases.

## 📊 Project Status

| Dataset | Target | Completed | Status |
|---------|--------|-----------|--------|
| **DrugBank** (Drugs) | 20,000 | 19,528 | ✅ Complete |
| **HPO** (Symptoms) | 20,000 | 20,000 | ✅ Complete |
| **BioASQ/PubMedQA** (CLTL) | 2,000 | 1,890 | ✅ Complete |
| **ICD-10 VN** (Diseases) | 20,000 | 24,438 | ✅ Complete |
| **ViMedAQA** (Reasoning) | 40,000 | ~40,000 | ✅ Complete |
| **Total** | **102,000** | **~105,856** | **100% Complete** |

## 🎯 Project Goals

1. **Vietnamese Medical NLP**: Create the largest Vietnamese medical QA dataset for SLM training
2. **Balanced Training Data**: Ensure 50/50 Đúng/Sai distribution for unbiased learning
3. **Multi-domain Coverage**: Drugs, symptoms, diseases, and medical reasoning
4. **Cross-lingual Transfer**: Bridge international medical knowledge to Vietnamese

## 📁 Project Structure

```
slm-for-disease-qa/
├── checkpoints/                       # Model Weights
│   ├── checkpoint-8000/               # 🏆 BEST MODEL (~74% Acc)
│   ├── checkpoint-8652/               # Latest checkpoint
│   └── final/                         # Final export (same as 3000)
├── data/                              # Data Directory
│   ├── sources/                       # Raw source data
│   │   ├── DrugBank/
│   │   ├── HPO/
│   │   ├── ICD10/
│   │   ├── ViMedAQA/
│   │   └── BioASQ14b/
│   └── processed/                     # Training ready datasets
│       └── train_data_v2/             # Unified dataset (Train/Val/Test)
├── src/                               # Source Code
│   ├── training/                      # Training scripts (Modal)
│   ├── evaluation/                    # Evaluation scripts
│   ├── data_generation/               # Dataset creation pipeline
│   └── processing/                    # Analysis & translation
└── results/                           # Evaluation logs & metrics
```

## 🧠 Model Checkpoints & Training Experiments

We conducted two training runs using different infrastructure.

| Experiment | Infrastructure | Best Accuracy | Checkpoint | Notes |
|------------|----------------|---------------|------------|-------|
| **Local Run** | **NVIDIA RTX 5080** | **74.83%** | *Local* | 🏆 **Best Overall Performance** |
| **Cloud Run** | Modal H100 | 74.16% | `checkpoint-8000` | Cloud baseline, available in repo |

### Cloud Checkpoints (Available in `checkpoints/`)

| Checkpoint | Training Step | Accuracy | Description |
|------------|---------------|----------|-------------|
| **checkpoint-8000** | 8000 | **74.16%** | Best Cloud Model |
| checkpoint-8652 | 8652 (Max) | 73.68% | Slight overfitting |
| checkpoint-3000 | 3000 | 64.85% | Early baseline |

## 📋 Dataset Formats

### Standard Vietnamese Medical QA Format (Alpaca)

Intermediate format used for processing:

```json
{
  "instruction": "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai.",
  "input": "Suy tim (Heart failure) có phải là một dạng của Rối loạn tim mạch (Cardiovascular abnormality) không?",
  "output": "Đúng"
}
```

### Final Training Format (Gemma Chat)

The model is trained on this conversational format:

```json
{
  "messages": [
    {"role": "user", "content": "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai.\nSuy tim...?"},
    {"role": "model", "content": "Đúng"}
  ]
}
```

## 🚀 Quick Start

### 1. Load the Best Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "google/gemma-1b-it"
adapter_path = "checkpoints/checkpoint-8000"

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
```

### 2. Using the Datasets

```python
import json

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load unified training set
train_set = load_jsonl('data/processed/train_data_v2/train.jsonl')
val_set = load_jsonl('data/processed/train_data_v2/val.jsonl')

print(f"Train: {len(train_set)}, Val: {len(val_set)}")
```

## 🔧 Training Pipeline
 
Training was performed in two environments to validate performance:
 
1.  **Cloud (Modal.com)**: NVIDIA H100 (80GB) - For scalability and reproducibility.
2.  **Local Workstation**: NVIDIA RTX 5080 (32GB) - Achieved highest accuracy (74.83%).
 
### Key Scripts
 
1. **Data Generation**: `src/data_generation/create_train_data_v2.py`
2. **Training**: `src/training/train_modal_resume.py`
3. **Evaluation**: `src/evaluation/eval_all_checkpoints.py`

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DrugBank**: Drug vocabulary and classifications
- **HPO (Human Phenotype Ontology)**: Symptom hierarchy and relationships
- **BioASQ**: Biomedical question answering challenge
- **PubMedQA**: PubMed-based QA dataset
- **Modal**: GPU cloud infrastructure
- **Bộ Y tế Việt Nam**: ICD-10 Data

---

**Created for Vietnamese Medical AI Research** 🇻🇳