#!/usr/bin/env python3
"""
Create balanced training batches for PQA-A dataset
Author: Generated for PubMedQA project
"""

import json
import random
import math
from collections import defaultdict

def create_balanced_training_data(yes_file, no_file, output_file, strategy='oversample_minority'):
    """
    Create balanced training data using different strategies
    
    Args:
        yes_file: Path to yes samples JSON
        no_file: Path to no samples JSON  
        output_file: Path to save balanced dataset
        strategy: 'oversample_minority', 'undersample_majority', or 'hybrid'
    """
    
    print("Loading datasets...")
    with open(yes_file, 'r') as f:
        yes_data = json.load(f)
    with open(no_file, 'r') as f:
        no_data = json.load(f)
    
    yes_count = len(yes_data)
    no_count = len(no_data)
    
    print(f"Original distribution:")
    print(f"  YES: {yes_count:,} samples")
    print(f"  NO: {no_count:,} samples")
    print(f"  Ratio: {yes_count/no_count:.1f}:1")
    
    balanced_data = {}
    final_yes = 0
    final_no = 0
    
    if strategy == 'oversample_minority':
        # Keep all majority class, oversample minority
        print(f"\nUsing oversampling strategy...")
        
        # Add all yes samples
        balanced_data.update(yes_data)
        
        # Oversample no samples to match yes count
        no_pmids = list(no_data.keys())
        target_no_count = yes_count
        
        for i in range(target_no_count):
            original_pmid = no_pmids[i % no_count]
            if i < no_count:
                # Original sample
                new_pmid = original_pmid
            else:
                # Oversampled copy with modified PMID
                new_pmid = f"{original_pmid}_dup_{i // no_count}"
            balanced_data[new_pmid] = no_data[original_pmid].copy()
        
        final_yes = yes_count
        final_no = target_no_count
    
    elif strategy == 'undersample_majority':
        # Downsample majority class to match minority
        print(f"\nUsing undersampling strategy...")
        
        # Randomly sample yes data to match no count
        yes_pmids = list(yes_data.keys())
        random.shuffle(yes_pmids)
        selected_yes = yes_pmids[:no_count]
        
        for pmid in selected_yes:
            balanced_data[pmid] = yes_data[pmid]
        
        # Add all no samples
        balanced_data.update(no_data)
        
        final_yes = no_count
        final_no = no_count
    
    elif strategy == 'hybrid':
        # Compromise: partial oversampling + partial undersampling
        print(f"\nUsing hybrid strategy...")
        
        # Target size somewhere in between
        target_size = int(math.sqrt(yes_count * no_count))  # Geometric mean
        
        # Undersample yes
        yes_pmids = list(yes_data.keys())
        random.shuffle(yes_pmids)
        selected_yes = yes_pmids[:target_size]
        
        for pmid in selected_yes:
            balanced_data[pmid] = yes_data[pmid]
        
        # Oversample no
        no_pmids = list(no_data.keys())
        for i in range(target_size):
            original_pmid = no_pmids[i % no_count]
            if i < no_count:
                new_pmid = original_pmid
            else:
                new_pmid = f"{original_pmid}_dup_{i // no_count}"
            balanced_data[new_pmid] = no_data[original_pmid].copy()
        
        final_yes = target_size
        final_no = target_size
    
    print(f"\nFinal balanced distribution:")
    print(f"  YES: {final_yes:,} samples")
    print(f"  NO: {final_no:,} samples") 
    print(f"  Total: {len(balanced_data):,} samples")
    print(f"  Ratio: 1:1")
    
    # Save balanced dataset
    with open(output_file, 'w') as f:
        json.dump(balanced_data, f, indent=4)
    
    print(f"\nBalanced dataset saved to: {output_file}")

def main():
    """Create balanced datasets using different strategies"""
    
    random.seed(42)  # For reproducibility
    
    base_dir = "../data"
    
    # Define input files
    train_yes = f"{base_dir}/pqaa_train_splits/yes.json"
    train_no = f"{base_dir}/pqaa_train_splits/no.json"
    
    # Create balanced training sets with different strategies
    strategies = [
        ('oversample_minority', 'pqaa_train_balanced_oversample.json'),
        ('undersample_majority', 'pqaa_train_balanced_undersample.json'),
        ('hybrid', 'pqaa_train_balanced_hybrid.json')
    ]
    
    for strategy, output_filename in strategies:
        print(f"\n{'='*60}")
        print(f"Creating balanced dataset with {strategy} strategy")
        print(f"{'='*60}")
        
        output_path = f"{base_dir}/{output_filename}"
        create_balanced_training_data(train_yes, train_no, output_path, strategy)

if __name__ == "__main__":
    main()