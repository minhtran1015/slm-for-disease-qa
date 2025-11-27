# SLM for Disease QA

**Vietnamese Medical Question-Answering Dataset for Small Language Models**

A comprehensive research project for training Small Language Models (SLMs) on medical question-answering tasks in Vietnamese. This project creates high-quality, balanced datasets from international medical ontologies and databases.

## 📊 Project Status

| Dataset | Target | Completed | Status |
|---------|--------|-----------|--------|
| **DrugBank** (Drugs) | 20,000 | 19,754 | ✅ Complete |
| **HPO** (Symptoms) | 20,000 | 20,000 | ✅ Complete |
| **BioASQ/PubMedQA** (CLTL) | 2,000 | 1,890 | ✅ Complete |
| **ICD-10 VN** (Diseases) | 20,000 | - | ⏳ Pending |
| **ViMedAQA** (Reasoning) | 40,000 | - | ⏳ Pending |
| **Total** | **102,000** | **41,644** | **41% Complete** |

## 🎯 Project Goals

1. **Vietnamese Medical NLP**: Create the largest Vietnamese medical QA dataset for SLM training
2. **Balanced Training Data**: Ensure 50/50 Đúng/Sai distribution for unbiased learning
3. **Multi-domain Coverage**: Drugs, symptoms, diseases, and medical reasoning
4. **Cross-lingual Transfer**: Bridge international medical knowledge to Vietnamese

## 📁 Dataset Structure

```
slm-for-disease-qa/
├── DrugBank/                          # Drug identification (19,754 samples)
│   ├── drugbank_qa_vietnamese_extended_train.jsonl
│   └── drugbank_qa_vietnamese_extended_test.jsonl
├── HPO/                               # Symptom relationships (20,000 samples)
│   ├── hpo_vietnamese_bilingual_train.jsonl    # Recommended
│   ├── hpo_vietnamese_bilingual_test.jsonl
│   └── convert_hpo_bilingual_modal.py
├── BioASQ14b/                         # Source data for CLTL
├── PubMedQA/                          # Source data for CLTL
├── medical_qa_vietnamese_cltl_*.jsonl # Cross-lingual (1,890 samples)
├── ICD10/                             # Diseases (pending)
└── ViMedAQA/                          # Medical reasoning (pending)
```

## 📋 Dataset Formats

### Standard Vietnamese Medical QA Format

All datasets follow this unified format:

```json
{
  "instruction": "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai.",
  "input": "Suy tim (Heart failure) có phải là một dạng của Rối loạn tim mạch (Cardiovascular abnormality) không?",
  "output": "Đúng"
}
```

### Dataset-Specific Features

#### 1. DrugBank (Drug Identification)

- **19,754 samples** (17,780 train + 1,974 test)
- Scientific drug names + Vietnamese common names
- 50% Đúng / 50% Sai balance

#### 2. HPO (Symptom Relationships) - Bilingual

- **20,000 samples** (18,000 train + 2,000 test)
- Vietnamese (English) bilingual format
- 8 instruction templates + 8 question templates
- GPU-accelerated translation via Modal + NLLB-200

#### 3. CLTL (Cross-Lingual Transfer Learning)

- **1,890 samples** (1,701 train + 189 test)
- English medical context + Vietnamese Q&A
- Sources: BioASQ 14b + PubMedQA labeled

## 🚀 Quick Start

### Using the Datasets

```python
import json

# Load any dataset
def load_jsonl(filepath):
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# Example: Load HPO bilingual dataset
train_data = load_jsonl('HPO/hpo_vietnamese_bilingual_train.jsonl')
test_data = load_jsonl('HPO/hpo_vietnamese_bilingual_test.jsonl')

print(f"Train: {len(train_data)}, Test: {len(test_data)}")
# Output: Train: 18000, Test: 2000
```

### Training Format

```python
# For SFT (Supervised Fine-Tuning)
for sample in train_data:
    prompt = f"{sample['instruction']}\n\n{sample['input']}"
    response = sample['output']  # "Đúng" or "Sai"
```

## 🔧 Generating New Data

### HPO Bilingual Dataset (Modal GPU)

```bash
# Install Modal
pip install modal
modal setup

# Upload source data
modal volume put medical-data HPO/hp.json hp.json

# Run translation pipeline
cd HPO/
modal run --detach convert_hpo_bilingual_modal.py

# Download results
modal volume get medical-data hpo_vietnamese_bilingual_train.jsonl ./
modal volume get medical-data hpo_vietnamese_bilingual_test.jsonl ./
```

### CLTL Dataset

```bash
# Upload source data
modal volume put medical-data BioASQ14b/training14b.json training14b.json
modal volume put medical-data PubMedQA/ori_pqal.json ori_pqal.json

# Run translation
modal run --detach translate_medical_crosslingual.py
```

## 📈 Template Variety

### Instruction Templates (8 variants)

- "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai."
- "Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa."
- "Xác định tính đúng sai của nhận định sau về triệu chứng y học."
- "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau."
- "Dựa vào phân loại triệu chứng y học, hãy trả lời Đúng hoặc Sai."
- "Với kiến thức về bệnh học, hãy xác nhận câu sau Đúng hay Sai."
- "Câu hỏi về mối quan hệ triệu chứng - Trả lời Đúng hoặc Sai."
- "Theo hệ thống phân loại y khoa, câu sau Đúng hay Sai?"

### Question Templates (8 variants per type)

- "... có phải là một dạng của ... không?"
- "Trong y học, ... thuộc nhóm ... đúng không?"
- "Triệu chứng ... có nằm trong nhóm ... không?"
- "... có liên quan đến ... không?"
- And more...

## 🏗️ Technical Architecture

### Translation Pipeline

- **Model**: facebook/nllb-200-distilled-600M
- **Infrastructure**: Modal A10G GPU
- **Batch Size**: 128 terms per batch
- **Processing Time**: ~2.5 minutes for 20k samples

### Data Quality

- **Balanced Classes**: 50% positive / 50% negative
- **Bilingual Backup**: Vietnamese (English) format preserves original terms
- **Diverse Templates**: Reduces model overfitting to specific patterns

## 📖 Documentation

- [Copilot Instructions](.github/copilot-instructions.md) - Detailed AI agent guidelines
- [DrugBank README](DrugBank/README_VIETNAMESE_EXTENDED.md) - Drug dataset documentation
- [HPO README](HPO/README.md) - Symptom dataset documentation

## 🔬 Research Applications

- Vietnamese medical chatbots
- Clinical decision support systems
- Medical education tools
- Drug information systems
- Symptom checking applications

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DrugBank**: Drug vocabulary and classifications
- **HPO (Human Phenotype Ontology)**: Symptom hierarchy and relationships
- **BioASQ**: Biomedical question answering challenge
- **PubMedQA**: PubMed-based QA dataset
- **Modal**: GPU cloud infrastructure for translation

---

**Created for Vietnamese Medical AI Research** 🇻🇳