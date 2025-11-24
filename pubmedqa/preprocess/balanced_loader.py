#!/usr/bin/env python3
"""
Balanced data loader utility for training LLMs on imbalanced PubMedQA data
Author: Generated for PubMedQA project
"""

import json
import random
import math
from typing import Iterator, Dict, List, Tuple
from collections import defaultdict

class BalancedPubMedQALoader:
    """
    Data loader that creates balanced mini-batches for training
    """
    
    def __init__(self, yes_file: str, no_file: str, batch_size: int = 32, 
                 balance_ratio: float = 0.5, shuffle: bool = True):
        """
        Args:
            yes_file: Path to JSON file with "yes" samples
            no_file: Path to JSON file with "no" samples  
            batch_size: Number of samples per batch
            balance_ratio: Fraction of batch that should be "yes" samples (0.5 = equal)
            shuffle: Whether to shuffle data
        """
        self.batch_size = batch_size
        self.balance_ratio = balance_ratio
        self.shuffle = shuffle
        
        # Load data
        print("Loading datasets...")
        with open(yes_file, 'r') as f:
            self.yes_data = json.load(f)
        with open(no_file, 'r') as f:
            self.no_data = json.load(f)
        
        # Convert to lists for easier sampling
        self.yes_samples = [(pmid, data) for pmid, data in self.yes_data.items()]
        self.no_samples = [(pmid, data) for pmid, data in self.no_data.items()]
        
        print(f"Loaded {len(self.yes_samples):,} YES samples")
        print(f"Loaded {len(self.no_samples):,} NO samples")
        
        # Calculate batch composition
        self.yes_per_batch = int(batch_size * balance_ratio)
        self.no_per_batch = batch_size - self.yes_per_batch
        
        print(f"Batch composition: {self.yes_per_batch} YES + {self.no_per_batch} NO = {batch_size} total")
        
    def __iter__(self) -> Iterator[List[Tuple[str, Dict]]]:
        """Generate balanced batches"""
        
        if self.shuffle:
            random.shuffle(self.yes_samples)
            random.shuffle(self.no_samples)
        
        # Calculate number of batches based on limiting factor
        max_yes_batches = len(self.yes_samples) // self.yes_per_batch if self.yes_per_batch > 0 else float('inf')
        max_no_batches = len(self.no_samples) // self.no_per_batch if self.no_per_batch > 0 else float('inf')
        
        num_batches = int(min(max_yes_batches, max_no_batches))
        
        print(f"Generating {num_batches} balanced batches...")
        
        for i in range(num_batches):
            batch = []
            
            # Add YES samples
            yes_start = i * self.yes_per_batch
            yes_end = yes_start + self.yes_per_batch
            batch.extend(self.yes_samples[yes_start:yes_end])
            
            # Add NO samples  
            no_start = i * self.no_per_batch
            no_end = no_start + self.no_per_batch
            batch.extend(self.no_samples[no_start:no_end])
            
            # Shuffle batch
            if self.shuffle:
                random.shuffle(batch)
                
            yield batch
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'yes_count': len(self.yes_samples),
            'no_count': len(self.no_samples),
            'total_count': len(self.yes_samples) + len(self.no_samples),
            'original_ratio': len(self.yes_samples) / len(self.no_samples),
            'batch_size': self.batch_size,
            'yes_per_batch': self.yes_per_batch,
            'no_per_batch': self.no_per_batch,
            'balance_ratio': self.balance_ratio
        }

def create_weighted_sampling_config():
    """Create configuration for weighted sampling (for PyTorch, etc.)"""
    
    # Load split data to get counts
    with open("../data/pqaa_train_splits/yes.json", 'r') as f:
        yes_data = json.load(f)
    with open("../data/pqaa_train_splits/no.json", 'r') as f:
        no_data = json.load(f)
    
    yes_count = len(yes_data)
    no_count = len(no_data)
    total_count = yes_count + no_count
    
    # Calculate class weights (inverse frequency)
    yes_weight = total_count / (2 * yes_count)
    no_weight = total_count / (2 * no_count)
    
    config = {
        "class_weights": {
            "yes": yes_weight,
            "no": no_weight
        },
        "sample_weights_description": "Use these for weighted sampling in PyTorch DataLoader",
        "focal_loss_alpha": no_count / total_count,  # For focal loss
        "statistics": {
            "yes_samples": yes_count,
            "no_samples": no_count,
            "imbalance_ratio": yes_count / no_count,
            "minority_class_percentage": (no_count / total_count) * 100
        }
    }
    
    return config

def demonstrate_balanced_loader():
    """Demonstrate the balanced data loader"""
    
    print("="*60)
    print("DEMONSTRATING BALANCED DATA LOADER")
    print("="*60)
    
    # Create loader
    loader = BalancedPubMedQALoader(
        yes_file="../data/pqaa_train_splits/yes.json",
        no_file="../data/pqaa_train_splits/no.json",
        batch_size=16,
        balance_ratio=0.5,  # 50-50 split
        shuffle=True
    )
    
    # Show statistics
    stats = loader.get_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}")
    
    # Show first few batches
    print(f"\nFirst 3 batches (showing answer distribution):")
    for i, batch in enumerate(loader):
        if i >= 3:
            break
            
        yes_count = sum(1 for _, data in batch if data['final_decision'] == 'yes')
        no_count = len(batch) - yes_count
        
        print(f"  Batch {i+1}: {yes_count} YES, {no_count} NO (total: {len(batch)})")

def main():
    """Main function"""
    
    # Demonstrate balanced loader
    demonstrate_balanced_loader()
    
    # Create weighted sampling config
    print(f"\n" + "="*60)
    print("WEIGHTED SAMPLING CONFIGURATION")
    print("="*60)
    
    config = create_weighted_sampling_config()
    
    # Save config
    config_path = "../data/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Training configuration saved to: {config_path}")
    print(f"\nKey recommendations:")
    print(f"  - Use class weights: YES={config['class_weights']['yes']:.3f}, NO={config['class_weights']['no']:.3f}")
    print(f"  - Minority class represents {config['statistics']['minority_class_percentage']:.1f}% of data")
    print(f"  - Consider focal loss with alpha={config['focal_loss_alpha']:.3f}")

if __name__ == "__main__":
    main()