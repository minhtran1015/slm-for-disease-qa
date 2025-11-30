# Scripts Directory

This folder contains all data processing and pipeline utilities for the SLM Disease QA project.

## Pipeline Scripts

### Core Processing
- **`unify_training_data.py`** - Merges all 5 datasets into unified training format (79,697 samples)
- **`convert_eval_data.py`** - Converts CSV evaluation data to JSONL format  
- **`validate_unified_data.py`** - Validates all training data integrity and format

### Data Standardization (Surface-Level Fixes)
- **`fix_hpo_instructions.py`** - Fixed HPO instruction inconsistencies
- **`fix_all_instructions.py`** - Applied final instruction standardization
- **`convert_vimedaqa_to_standard.py`** - Converted ViMedAQA from chat to instruction format

### Cross-Lingual Processing
- **`translate_medical_crosslingual.py`** - Generates CLTL dataset (BioASQ + PubMedQA → Vietnamese)

## Usage

All scripts are designed to run from this `scripts/` directory:

```bash
cd scripts/

# Regenerate unified training data
python unify_training_data.py

# Convert evaluation data to JSONL
python convert_eval_data.py

# Validate all data files
python validate_unified_data.py
```

## Output

Scripts output data to the following locations:
- Training data: `../train_data/`
- Individual datasets: `../DrugBank/`, `../HPO/`, etc.
- Input data: `../data/`

All scripts use relative paths and must be run from the `scripts/` directory.