#!/usr/bin/env python3
"""
Split existing PQA-A train/dev sets by answer types
Author: Generated for PubMedQA project
"""

import json
import os
import sys
from collections import Counter

def split_existing_sets():
    """
    Split the existing pqaa_train_set.json and pqaa_dev_set.json by answer types
    """
    
    base_dir = "../data"
    files_to_process = [
        ("pqaa_train_set.json", "pqaa_train_splits"),
        ("pqaa_dev_set.json", "pqaa_dev_splits")
    ]
    
    for input_filename, output_dirname in files_to_process:
        input_path = os.path.join(base_dir, input_filename)
        output_dir = os.path.join(base_dir, output_dirname)
        
        if not os.path.exists(input_path):
            print(f"Skipping {input_filename} - file not found")
            continue
            
        print(f"\nProcessing {input_filename}...")
        
        # Load dataset
        with open(input_path, 'r') as f:
            dataset = json.load(f)
        
        # Initialize containers for different answer types
        answer_splits = {
            'yes': {},
            'no': {},
            'maybe': {},
            'others': {}
        }
        
        # Count distribution
        answer_counts = Counter()
        
        for pmid, data in dataset.items():
            final_decision = data.get('final_decision', 'unknown')
            answer_counts[final_decision] += 1
            
            if final_decision in ['yes', 'no', 'maybe']:
                answer_splits[final_decision][pmid] = data
            else:
                answer_splits['others'][pmid] = data
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits and report statistics
        total_samples = len(dataset)
        print(f"Total samples: {total_samples:,}")
        
        for answer_type, data in answer_splits.items():
            if data:  # Only save non-empty splits
                output_file = os.path.join(output_dir, f'{answer_type}.json')
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=4)
                
                count = len(data)
                percentage = (count / total_samples) * 100
                print(f"  {answer_type.upper()}: {count:,} samples ({percentage:.1f}%) -> {output_file}")
        
        # Answer distribution
        print(f"Answer distribution:")
        for answer, count in sorted(answer_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"  '{answer}': {count:,} ({percentage:.1f}%)")

def main():
    print("Splitting existing PQA-A train/dev sets by answer types...")
    
    try:
        split_existing_sets()
        print(f"\nSplitting completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()