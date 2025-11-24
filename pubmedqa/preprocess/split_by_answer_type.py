#!/usr/bin/env python3
"""
Split PQA-A dataset by answer types (yes/no/maybe)
Author: Generated for PubMedQA project
"""

import json
import os
import sys
from collections import Counter

def split_by_answer_type(input_file, output_dir):
    """
    Split the dataset by final_decision values
    
    Args:
        input_file: Path to input JSON file (e.g., ori_pqaa.json)
        output_dir: Directory to save split files
    """
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    # Initialize containers for different answer types
    answer_splits = {
        'yes': {},
        'no': {},
        'maybe': {},
        'others': {}  # For any unexpected values
    }
    
    # Count distribution for reporting
    answer_counts = Counter()
    
    print("Splitting by answer types...")
    for pmid, data in dataset.items():
        final_decision = data.get('final_decision', 'unknown')
        answer_counts[final_decision] += 1
        
        if final_decision in ['yes', 'no', 'maybe']:
            answer_splits[final_decision][pmid] = data
        else:
            answer_splits['others'][pmid] = data
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits and report statistics
    print("\nSaving splits and statistics:")
    total_samples = len(dataset)
    
    for answer_type, data in answer_splits.items():
        if data:  # Only save non-empty splits
            output_file = os.path.join(output_dir, f'pqaa_{answer_type}.json')
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            count = len(data)
            percentage = (count / total_samples) * 100
            print(f"  {answer_type.upper()}: {count:,} samples ({percentage:.1f}%) -> {output_file}")
        else:
            print(f"  {answer_type.upper()}: 0 samples (skipping)")
    
    print(f"\nTotal samples: {total_samples:,}")
    
    # Detailed statistics
    print(f"\nAnswer distribution:")
    for answer, count in sorted(answer_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"  '{answer}': {count:,} ({percentage:.1f}%)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python split_by_answer_type.py <input_file> <output_dir>")
        print("Example: python split_by_answer_type.py ../data/ori_pqaa.json ../data/pqaa_splits/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    try:
        split_by_answer_type(input_file, output_dir)
        print(f"\nSplitting completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()