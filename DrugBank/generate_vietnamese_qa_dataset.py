#!/usr/bin/env python3
"""
DrugBank QA Dataset Generator for Small Language Models (Gemma 1B, Qwen 0.5B)

This script processes the DrugBank vocabulary CSV file to generate 20k balanced
question-answer pairs suitable for training small language models on drug identification tasks.

Features:
- Balanced positive/negative samples (50/50 split)
- Diverse question templates to improve model generalization
- Multiple output formats (JSONL for training, JSON for analysis)
- Robust error handling and progress tracking
- Compatible with Hugging Face datasets format
"""

import json
import random
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

# ================= CONFIGURATION =================
DEFAULT_CONFIG = {
    'input_file': 'drugbank vocabulary.csv',
    'output_file': 'drugbank_qa_vietnamese_20k.jsonl',
    'analysis_file': 'drugbank_qa_vietnamese_analysis.json',
    'target_total': 20000,
    'seed': 42,
    'test_split_ratio': 0.1,  # 10% for test set
    'log_level': 'INFO'
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= VIETNAMESE QUESTION TEMPLATES =================
# Diverse templates for positive samples (drug-related questions in Vietnamese)
POSITIVE_TEMPLATES = [
    "{name} có phải là một loại thuốc hoặc dược phẩm không?",
    "Trong y học, {name} được phân loại là thuốc điều trị đúng không?",
    "Có phải {name} là tên một loại thuốc trong cơ sở dữ liệu y khoa?",
    "{name} có thể được kê đơn như một loại thuốc điều trị không?",
    "{name} là một chất dược phẩm dùng cho mục đích y tế phải không?",
    "{name} có đại diện cho một hoạt chất dược phẩm không?",
    "{name} có được liệt kê như một hợp chất thuốc điều trị không?",
    "Bạn có thể tìm thấy {name} trong tài liệu tham khảo dược lý không?",
    "{name} là một thực thể hóa học được sử dụng trong liệu pháp thuốc không?",
    "{name} có thuộc về danh mục thuốc y tế không?"
]

# Diverse templates for negative samples (non-drug entities in Vietnamese)
NEGATIVE_TEMPLATES = [
    "{name} có phải là một loại bệnh truyền nhiễm hoặc tình trạng bệnh lý không?",
    "{name} có đề cập đến một cấu trúc giải phẫu hoặc cơ quan của con người không?",
    "{name} là một triệu chứng y tế hoặc biểu hiện lâm sàng phải không?",
    "{name} có đại diện cho một thủ thuật phẫu thuật hoặc kỹ thuật y tế không?",
    "{name} có phải là một loại virus, vi khuẩn hoặc sinh vật gây bệnh không?",
    "{name} có đề cập đến một xét nghiệm chẩn đoán hoặc thủ thuật phòng thí nghiệm không?",
    "{name} là một thiết bị y tế hoặc dụng cụ phẫu thuật phải không?",
    "{name} có đại diện cho một rối loạn di truyền hoặc tình trạng di truyền không?",
    "{name} có phải là một loại ung thư hoặc khối u ác tính không?",
    "{name} có đề cập đến một quá trình sinh lý hoặc chức năng cơ thể không?"
]

# Additional context for instruction following in Vietnamese
INSTRUCTION_TEMPLATES = [
    "Dựa vào kiến thức dược học, hãy trả lời câu hỏi sau đây bằng Đúng hoặc Sai.",
    "Xác định xem thực thể đã cho có phải là thuốc hay dược phẩm không. Trả lời bằng Đúng hoặc Sai.",
    "Sử dụng kiến thức y khoa của bạn, phân loại xem đây có phải là hợp chất dược phẩm không. Trả lời Đúng hoặc Sai.",
    "Đánh giá câu hỏi sau về phân loại thuốc và trả lời bằng Đúng hoặc Sai.",
    "Dựa trên cơ sở dữ liệu y khoa và thông tin dược phẩm, hãy trả lời Đúng hoặc Sai."
]


class DrugBankQAGenerator:
    """Generator for DrugBank-based QA dataset suitable for small language models."""
    
    def __init__(self, config: Dict = None):
        """Initialize the generator with configuration parameters."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        random.seed(self.config['seed'])
        
        # Statistics tracking
        self.stats = {
            'total_drugs_available': 0,
            'drugs_used': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'total_generated': 0
        }
    
    def load_drugbank_data(self) -> List[str]:
        """
        Load and clean drug names from DrugBank CSV file.
        
        Returns:
            List of unique, cleaned drug names
        """
        logger.info(f"Loading DrugBank data from: {self.config['input_file']}")
        
        try:
            df = pd.read_csv(self.config['input_file'])
            logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Extract drug names from 'Common name' column
            if 'Common name' not in df.columns:
                raise ValueError("'Common name' column not found in CSV file")
            
            # Clean and filter drug names
            drug_names = df['Common name'].dropna().astype(str).tolist()
            
            # Additional cleaning: remove very short names and special characters
            cleaned_drugs = []
            for drug in drug_names:
                drug = drug.strip()
                # Filter out very short names (likely abbreviations) and special cases
                if len(drug) >= 3 and not drug.startswith('DB') and drug.isalnum() == False:
                    # Keep drugs with letters and some special chars, but filter pure numbers
                    if any(c.isalpha() for c in drug):
                        cleaned_drugs.append(drug)
            
            # Remove duplicates while preserving order
            unique_drugs = list(dict.fromkeys(cleaned_drugs))
            
            self.stats['total_drugs_available'] = len(unique_drugs)
            logger.info(f"Extracted {len(unique_drugs)} unique drug names after cleaning")
            
            return unique_drugs
            
        except Exception as e:
            logger.error(f"Error loading DrugBank data: {e}")
            # Fallback to mock data for testing
            logger.warning("Generating mock data for testing purposes")
            return self._generate_mock_drugs()
    
    def _generate_mock_drugs(self) -> List[str]:
        """Generate mock drug names for testing when real data isn't available."""
        common_drugs = [
            "Acetaminophen", "Ibuprofen", "Aspirin", "Metformin", "Lisinopril",
            "Amlodipine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan",
            "Albuterol", "Gabapentin", "Hydrochlorothiazide", "Sertraline", "Montelukast"
        ]
        
        # Generate additional mock names to ensure we have enough
        mock_drugs = common_drugs.copy()
        for i in range(15000):  # Generate enough for 20k samples
            mock_drugs.append(f"MockDrug_{i:04d}")
        
        return mock_drugs
    
    def generate_qa_pair(self, drug_name: str, is_positive: bool) -> Dict:
        """
        Generate a single question-answer pair.
        
        Args:
            drug_name: Name of the drug to use in the question
            is_positive: Whether to generate a positive (True) or negative (False) sample
            
        Returns:
            Dictionary containing the QA pair in training format
        """
        # Select appropriate template
        if is_positive:
            question_template = random.choice(POSITIVE_TEMPLATES)
            answer = "Đúng"
        else:
            question_template = random.choice(NEGATIVE_TEMPLATES)
            answer = "Sai"
        
        # Format the question
        question = question_template.format(name=drug_name)
        
        # Select instruction template
        instruction = random.choice(INSTRUCTION_TEMPLATES)
        
        return {
            "instruction": instruction,
            "input": question,
            "output": answer,
            "drug_name": drug_name,
            "sample_type": "positive" if is_positive else "negative"
        }
    
    def generate_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate the complete QA dataset.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        logger.info("Starting dataset generation...")
        
        # Load drug data
        all_drugs = self.load_drugbank_data()
        
        if not all_drugs:
            raise ValueError("No drug data available for generation")
        
        # Calculate how many drugs we need (each drug generates 2 samples: 1 positive, 1 negative)
        target_samples = self.config['target_total']
        drugs_needed = target_samples // 2
        
        if len(all_drugs) < drugs_needed:
            logger.warning(f"Not enough drugs ({len(all_drugs)}) for target samples ({target_samples}). "
                          f"Will generate {len(all_drugs) * 2} samples instead.")
            selected_drugs = all_drugs
        else:
            selected_drugs = random.sample(all_drugs, drugs_needed)
        
        self.stats['drugs_used'] = len(selected_drugs)
        
        # Generate QA pairs
        all_samples = []
        logger.info(f"Generating QA pairs from {len(selected_drugs)} drugs...")
        
        for drug in tqdm(selected_drugs, desc="Generating samples"):
            # Generate one positive and one negative sample per drug
            positive_sample = self.generate_qa_pair(drug, is_positive=True)
            negative_sample = self.generate_qa_pair(drug, is_positive=False)
            
            all_samples.extend([positive_sample, negative_sample])
            self.stats['positive_samples'] += 1
            self.stats['negative_samples'] += 1
        
        # Shuffle all samples
        random.shuffle(all_samples)
        self.stats['total_generated'] = len(all_samples)
        
        # Split into train/test
        test_size = int(len(all_samples) * self.config['test_split_ratio'])
        test_data = all_samples[:test_size]
        train_data = all_samples[test_size:]
        
        logger.info(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")
        
        return train_data, test_data
    
    def save_dataset(self, train_data: List[Dict], test_data: List[Dict]) -> None:
        """Save the generated dataset to files."""
        
        # Save training data in JSONL format (standard for LLM training)
        train_file = self.config['output_file']
        logger.info(f"Saving training data to: {train_file}")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_data:
                # Remove metadata fields for training
                training_sample = {
                    "instruction": sample["instruction"],
                    "input": sample["input"],
                    "output": sample["output"]
                }
                json.dump(training_sample, f, ensure_ascii=False)
                f.write('\n')
        
        # Save test data
        test_file = train_file.replace('.jsonl', '_test.jsonl')
        logger.info(f"Saving test data to: {test_file}")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in test_data:
                training_sample = {
                    "instruction": sample["instruction"],
                    "input": sample["input"],
                    "output": sample["output"]
                }
                json.dump(training_sample, f, ensure_ascii=False)
                f.write('\n')
        
        # Save analysis and statistics
        analysis_data = {
            "config": self.config,
            "statistics": self.stats,
            "sample_distribution": {
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "total_samples": len(train_data) + len(test_data),
                "positive_ratio": self.stats['positive_samples'] / self.stats['total_generated'],
                "negative_ratio": self.stats['negative_samples'] / self.stats['total_generated']
            },
            "sample_examples": {
                "positive_example": next((s for s in train_data if s["sample_type"] == "positive"), None),
                "negative_example": next((s for s in train_data if s["sample_type"] == "negative"), None)
            }
        }
        
        analysis_file = self.config['analysis_file']
        logger.info(f"Saving analysis to: {analysis_file}")
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    def print_statistics(self) -> None:
        """Print generation statistics."""
        print("\n" + "="*60)
        print("DRUGBANK QA DATASET GENERATION COMPLETE")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   • Total drugs available: {self.stats['total_drugs_available']:,}")
        print(f"   • Drugs used for generation: {self.stats['drugs_used']:,}")
        print(f"   • Positive samples (True): {self.stats['positive_samples']:,}")
        print(f"   • Negative samples (False): {self.stats['negative_samples']:,}")
        print(f"   • Total samples generated: {self.stats['total_generated']:,}")
        print(f"   • Target samples: {self.config['target_total']:,}")
        print(f"\n📁 Output files:")
        print(f"   • Training data: {self.config['output_file']}")
        print(f"   • Test data: {self.config['output_file'].replace('.jsonl', '_test.jsonl')}")
        print(f"   • Analysis: {self.config['analysis_file']}")
        print("\n✅ Dataset ready for training with Gemma 1B or Qwen 0.5B!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate DrugBank QA dataset for SLM training')
    parser.add_argument('--input', default=DEFAULT_CONFIG['input_file'], 
                       help='Input DrugBank CSV file')
    parser.add_argument('--output', default=DEFAULT_CONFIG['output_file'],
                       help='Output JSONL file for training data')
    parser.add_argument('--samples', type=int, default=DEFAULT_CONFIG['target_total'],
                       help='Target number of samples to generate')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        'input_file': args.input,
        'output_file': args.output,
        'target_total': args.samples,
        'seed': args.seed
    })
    
    try:
        # Initialize generator
        generator = DrugBankQAGenerator(config)
        
        # Generate dataset
        train_data, test_data = generator.generate_dataset()
        
        # Save results
        generator.save_dataset(train_data, test_data)
        
        # Print statistics
        generator.print_statistics()
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()