#!/usr/bin/env python3
"""
ICD-10 Yes/No Question Generator

This script generates Yes/No questions for medical training using ICD-10 disease codes.
- For each disease code, generates 2 questions:
  1. YES answer: "Mã [Code] là bệnh [Tên]?" (correct pairing)
  2. NO answer: "Mã [Code] là bệnh [Wrong_Name]?" (incorrect pairing)
"""

import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Tuple

# STANDARDIZED INSTRUCTION - Force consistent user instruction following
# Critical: Use Đúng/Sai (not Có/Không) to match unified training format
STANDARD_INSTRUCTION_PREFIX = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "

def load_disease_data(json_file: str) -> List[Dict]:
    """Load disease data from JSON file."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['diseases']

def generate_yes_no_questions(diseases: List[Dict], num_pairs_per_disease: int = 1) -> List[Dict]:
    """
    Generate Yes/No questions for ICD-10 diseases.
    
    CRITICAL FIX: 
    - Use standardized instruction prefix for ALL questions
    - Convert answers to Đúng/Sai (not Có/Không) for unified training
    - Remove conflicting answer instructions
    
    Args:
        diseases: List of disease dictionaries
        num_pairs_per_disease: Number of question pairs per disease
    
    Returns:
        List of question dictionaries
    """
    
    print(f"🔍 Generating Yes/No questions for {len(diseases)} diseases...")
    
    questions = []
    
    # Create a list of all Vietnamese names for random selection
    all_vietnamese_names = [d['vietnamese_name'] for d in diseases if d['vietnamese_name']]
    
    for i, disease in enumerate(diseases):
        code_no_dots = disease['code_without_dots']
        correct_name = disease['vietnamese_name']
        
        # Skip if essential data is missing
        if not code_no_dots or not correct_name:
            continue
        
        for pair_num in range(num_pairs_per_disease):
            # Generate YES question (correct pairing) - with standardized prefix
            yes_question = {
                "instruction": STANDARD_INSTRUCTION_PREFIX + f"Mã {code_no_dots} là bệnh {correct_name}.",
                "input": "",
                "output": "Đúng",  # CRITICAL: Use Đúng (not Có) for consistency
                "code": code_no_dots,
                "correct_name": correct_name,
                "question_type": "correct_pairing",
                "disease_id": f"{code_no_dots}_{pair_num}_yes"
            }
            questions.append(yes_question)
            
            # Generate NO question (incorrect pairing) - with standardized prefix
            # Select a random wrong Vietnamese name
            wrong_names = [name for name in all_vietnamese_names if name != correct_name]
            if wrong_names:
                wrong_name = random.choice(wrong_names)
                
                no_question = {
                    "instruction": STANDARD_INSTRUCTION_PREFIX + f"Mã {code_no_dots} là bệnh {wrong_name}.",
                    "input": "",
                    "output": "Sai",  # CRITICAL: Use Sai (not Không) for consistency
                    "code": code_no_dots,
                    "correct_name": correct_name,
                    "wrong_name": wrong_name,
                    "question_type": "incorrect_pairing",
                    "disease_id": f"{code_no_dots}_{pair_num}_no"
                }
                questions.append(no_question)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  📊 Generated questions for {i + 1} diseases...")
    
    print(f"✅ Generated {len(questions)} questions total (all with standardized prefix & Đúng/Sai)")
    return questions

def generate_gemma_chat_format(questions: List[Dict]) -> List[Dict]:
    """Convert questions to Gemma-1B chat format (NOT USED - questions already have standardized format)."""
    
    print("⏭️  Skipping: Questions already have standardized instruction format")
    
    # Keep questions as-is since they already have instruction + output format
    return questions

def save_questions_multiple_formats(questions: List[Dict], base_filename: str = "icd10_yesno_questions"):
    """Save questions in multiple formats (JSONL for training, JSON for analysis)."""
    
    output_dir = Path("generated_questions")
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON (full format with metadata)
    json_file = output_dir / f"{base_filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON saved to: {json_file}")
    
    # Save as JSONL for training (standardized instruction+output format)
    jsonl_file = output_dir / f"{base_filename}.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for question in questions:
            # Extract only the fields needed for training
            training_sample = {
                "instruction": question["instruction"],
                "input": question.get("input", ""),
                "output": question["output"]
            }
            f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
    print(f"📝 JSONL saved to: {jsonl_file}")
    
    # Save as CSV for analysis
    csv_file = output_dir / f"{base_filename}.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        if questions:
            fieldnames = ['instruction', 'output', 'code', 'correct_name', 
                         'question_type', 'disease_id']
            if 'wrong_name' in questions[0]:
                fieldnames.append('wrong_name')
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for question in questions:
                # Only include relevant fields for CSV
                csv_row = {k: v for k, v in question.items() if k in fieldnames}
                writer.writerow(csv_row)
    
    print(f"📊 CSV saved to: {csv_file}")
    
    return json_file, jsonl_file, csv_file

def create_balanced_dataset(questions: List[Dict]) -> List[Dict]:
    """Create a balanced dataset with equal Đúng/Sai questions (standardized format)."""
    
    # Use 'output' field for standardized format (Đúng/Sai)
    yes_questions = [q for q in questions if q['output'] == 'Đúng']
    no_questions = [q for q in questions if q['output'] == 'Sai']
    
    print(f"📊 Original distribution: {len(yes_questions)} Đúng, {len(no_questions)} Sai")
    
    # Take equal numbers of each
    min_count = min(len(yes_questions), len(no_questions))
    
    balanced_questions = (
        random.sample(yes_questions, min_count) + 
        random.sample(no_questions, min_count)
    )
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_questions)
    
    print(f"📊 Balanced distribution: {min_count} Đúng, {min_count} Sai")
    return balanced_questions

def generate_statistics(questions: List[Dict]) -> Dict:
    """Generate statistics about the generated questions."""
    
    stats = {
        "total_questions": len(questions),
        "yes_questions": len([q for q in questions if q['output'] == 'Đúng']),
        "no_questions": len([q for q in questions if q['output'] == 'Sai']),
        "unique_codes": len(set(q['code'] for q in questions)),
        "unique_diseases": len(set(q['correct_name'] for q in questions)),
        "question_types": {}
    }
    
    # Count question types
    for question in questions:
        qtype = question['question_type']
        stats['question_types'][qtype] = stats['question_types'].get(qtype, 0) + 1
    
    return stats

def print_sample_questions(questions: List[Dict], num_samples: int = 5):
    """Print sample questions for verification."""
    
    print(f"\n🔍 SAMPLE QUESTIONS:")
    print("=" * 60)
    
    # Show mix of Đúng and Sai questions
    yes_samples = [q for q in questions if q['output'] == 'Đúng'][:num_samples//2 + 1]
    no_samples = [q for q in questions if q['output'] == 'Sai'][:num_samples//2 + 1]
    
    samples = yes_samples + no_samples
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    for i, question in enumerate(samples, 1):
        print(f"{i}. Instruction: {question['instruction'][:80]}...")
        print(f"   Output: {question['output']}")
        print(f"   Code: {question['code']} | Type: {question['question_type']}")
        if 'wrong_name' in question:
            print(f"   Wrong: {question['wrong_name']}")
        print()

def main():
    """Main execution function."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("🚀 Starting ICD-10 Yes/No Question Generation")
    
    # Load disease data
    disease_file = "extracted_data/icd10_diseases_simple.json"
    
    if not Path(disease_file).exists():
        print(f"❌ Error: Disease data file '{disease_file}' not found!")
        print("Please run extract_icd10_simple.py first.")
        return
    
    diseases = load_disease_data(disease_file)
    print(f"📚 Loaded {len(diseases)} diseases")
    
    # Generate questions
    questions = generate_yes_no_questions(diseases, num_pairs_per_disease=1)
    
    # Convert to Gemma chat format
    gemma_questions = generate_gemma_chat_format(questions)
    
    # Create balanced dataset
    balanced_questions = create_balanced_dataset(gemma_questions)
    
    # Generate statistics
    stats = generate_statistics(gemma_questions)
    stats_balanced = generate_statistics(balanced_questions)
    
    # Save full dataset
    print("\n💾 Saving full dataset...")
    save_questions_multiple_formats(gemma_questions, "icd10_yesno_full")
    
    # Save balanced dataset
    print("\n💾 Saving balanced dataset...")
    save_questions_multiple_formats(balanced_questions, "icd10_yesno_balanced")
    
    # Save statistics
    output_dir = Path("generated_questions")
    stats_file = output_dir / "generation_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "full_dataset": stats,
            "balanced_dataset": stats_balanced,
            "generation_date": "2025-11-26",
            "random_seed": 42
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Statistics saved to: {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📋 GENERATION SUMMARY")
    print("="*60)
    print(f"📈 Total Questions (Full): {stats['total_questions']:,}")
    print(f"✅ YES Questions: {stats['yes_questions']:,}")
    print(f"❌ NO Questions: {stats['no_questions']:,}")
    print(f"🔢 Unique Disease Codes: {stats['unique_codes']:,}")
    print(f"🏥 Unique Disease Names: {stats['unique_diseases']:,}")
    
    print(f"\n📊 Balanced Dataset: {stats_balanced['total_questions']:,} questions")
    print(f"   ✅ YES: {stats_balanced['yes_questions']:,}")
    print(f"   ❌ NO: {stats_balanced['no_questions']:,}")
    
    # Show sample questions
    print_sample_questions(balanced_questions)
    
    print(f"\n✅ Question generation completed!")
    print(f"📂 All files saved in 'generated_questions/' directory")
    print(f"🤖 Ready for Gemma-1B training with chat format!")

if __name__ == "__main__":
    main()