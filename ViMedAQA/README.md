# ViMedAQA Yes/No Dataset Processing

This folder contains scripts to process the ViMedAQA dataset from Hugging Face to create balanced Yes/No question-answering datasets in Vietnamese for medical domain.

## Processing Methods

### Method 1: Simple Negative Sampling (`process_vimedaqa_yesno.py`)

The original script converts ViMedAQA Q&A pairs into a Yes/No format by:

1. Creating **positive samples**: Original question + correct answer → "Đúng" (True)
2. Creating **negative samples**: Original question + random wrong answer from same category → "Sai" (False)

### Method 2: Gemini API Statement Generation (`process_vimedaqa_gemini.py`) ⭐ NEW

Advanced pipeline using Google Gemini API to transform Q&A pairs into natural medical statements:

**Input (Q&A format):**

```text
Q: Paracetamol có phải là một loại thuốc giảm đau không?
A: Có, Paracetamol là thuốc giảm đau hạ sốt phổ biến.
```

**Output (Statement format):**

```text
Paracetamol là một loại thuốc có tác dụng giảm đau và hạ sốt. → Đúng
Paracetamol là thuốc kháng sinh điều trị nhiễm khuẩn. → Sai
```

This creates more natural medical statements similar to:

- "Ho kéo dài trên 3 tuần có phải là triệu chứng của lao phổi."
- "Sỏi thận hình thành do khoáng chất kết tụ trong nước tiểu."
- "Động kinh là tình trạng các tế bào não hoạt động bất thường gây co giật."

Both approaches create balanced binary classification datasets suitable for training Small Language Models (SLMs) on medical question verification tasks.

## Features

- **Balanced Dataset**: Equal numbers of positive and negative samples
- **Category-aware Negative Sampling**: Wrong answers come from the same medical category for harder learning
- **Vietnamese Language Support**: Optimized templates for Vietnamese medical Q&A
- **JSONL Format**: Ready for training with popular ML frameworks
- **Checkpoint Support**: Resume interrupted processing (Gemini pipeline)

## Usage

### Method 1: Simple Negative Sampling

```bash
pip install -r requirements.txt
python process_vimedaqa_yesno.py
```

### Method 2: Gemini API Pipeline

```bash
# Set your Gemini API key
export GEMINI_API_KEY='your-api-key-here'

# Run the pipeline
python process_vimedaqa_gemini.py
```

The Gemini pipeline features:

- **Periodic saving**: Saves checkpoint every 50 samples
- **Resume support**: Can resume from interruption
- **Rate limiting**: Respects API rate limits (15 req/min)
- **Balanced output**: Generates both TRUE and FALSE statements

### Output Format

Both scripts generate JSONL files compatible with LLM training:

```json
{
  "instruction": "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau.",
  "input": "Paracetamol là một loại thuốc có tác dụng giảm đau và hạ sốt",
  "output": "Đúng",
  "question_type": "vimedaqa_true",
  "source_question": "Paracetamol có phải là một loại thuốc giảm đau không?",
  "source_answer": "Có, Paracetamol là thuốc giảm đau hạ sốt phổ biến."
}
```

## Configuration

### Method 1 (`process_vimedaqa_yesno.py`)

- `OUTPUT_FILE`: Output filename (default: `vimedaqa_yesno_50k.jsonl`)
- `SUBSETS`: ViMedAQA categories (default: `['disease', 'drug', 'body-part']`)
- `SEED`: Random seed for reproducibility (default: `42`)

### Method 2 (`process_vimedaqa_gemini.py`)

- `GEMINI_MODEL`: Model to use (default: `gemini-2.0-flash`)
- `MAX_RETRIES`: API retry attempts (default: `3`)
- `REQUESTS_PER_MINUTE`: Rate limiting (default: `15`)
- `SAVE_INTERVAL`: Checkpoint frequency (default: `50`)
- `MAX_SAMPLES`: Limit samples for testing (default: `None` = all)

## File Structure

```text
ViMedAQA/
├── process_vimedaqa_yesno.py           # Simple negative sampling
├── process_vimedaqa_gemini.py          # Gemini API pipeline
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── train-00000-of-00001.parquet        # Source data (training)
├── test-00000-of-00001.parquet         # Source data (test)
├── validation-00000-of-00001.parquet   # Source data (validation)
└── vimedaqa_yesno_gemini_train.jsonl   # Output (after running Gemini pipeline)
```

## Integration with Main Project

This processed dataset contributes to the **40k ViMedAQA samples** target in the main project's data strategy for Vietnamese medical reasoning.
