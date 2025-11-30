# Data Directory

This folder contains the original evaluation datasets and processed cross-lingual data.

## Files

### Evaluation Data (Original CSV Format)
- **`dev_v0.5.csv`** - Development/validation set (623 samples)
- **`test_v0.5.csv`** - Test set (623 samples)  
- **`Dev_sample.v1.0.csv`** - Alternative development sample

### Cross-Lingual Transfer Learning (CLTL)
- **`medical_qa_vietnamese_cltl_train.jsonl`** - Training set (1,701 samples)
- **`medical_qa_vietnamese_cltl_test.jsonl`** - Test set (189 samples)
- **`medical_qa_vietnamese_cltl_stats.json`** - Dataset statistics

## Format

CSV files use TRUE/FALSE format and are converted to JSONL "Đúng/Sai" format by the conversion scripts.

CLTL files use the standardized instruction format with English medical context preserved for cross-lingual learning.