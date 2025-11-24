# ViMedAQA Yes/No Dataset Processing

This script processes the ViMedAQA dataset from Hugging Face to create a balanced Yes/No question-answering dataset in Vietnamese for medical domain.

## Overview

The script converts the original ViMedAQA Q&A pairs into a Yes/No format by:

1. Creating **positive samples**: Original question + correct answer → "Đúng" (True)
2. Creating **negative samples**: Original question + random wrong answer from same category → "Sai" (False)

This approach creates a balanced binary classification dataset suitable for training Small Language Models (SLMs) on medical question verification tasks.

## Features

- **Balanced Dataset**: Equal numbers of positive and negative samples
- **Category-aware Negative Sampling**: Wrong answers come from the same medical category for harder learning
- **Vietnamese Language Support**: Optimized templates for Vietnamese medical Q&A
- **JSONL Format**: Ready for training with popular ML frameworks

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python process_vimedaqa_yesno.py
```

### 3. Output

The script generates `vimedaqa_yesno_50k.jsonl` with the following format:

```json
{
  "instruction": "Dựa vào thông tin tham khảo, câu trả lời trên là ĐÚNG hay SAI so với câu hỏi?",
  "input": "Câu hỏi: [original question]\nThông tin tham khảo: [proposed answer]",
  "output": "Đúng"
}
```

## Configuration

You can modify these parameters in the script:

- `OUTPUT_FILE`: Output filename (default: `vimedaqa_yesno_50k.jsonl`)
- `SUBSETS`: ViMedAQA categories to process (default: `['disease', 'drug', 'body-part']`)
- `SEED`: Random seed for reproducibility (default: `42`)

## Dataset Statistics

The script will display:

- Total number of samples generated
- Positive vs negative sample counts
- Balance ratio
- Sample preview

## Integration with Main Project

This processed dataset can be used alongside the PubMedQA datasets in the main project for:

- **Cross-lingual medical Q&A**: Vietnamese medical knowledge verification
- **Domain adaptation**: Training models on Vietnamese medical terminology
- **Balanced learning**: Addressing class imbalance issues common in medical datasets

## File Structure

```text
ViMedAQA/
├── process_vimedaqa_yesno.py    # Main processing script
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── vimedaqa_yesno_50k.jsonl    # Generated dataset (after running script)
```