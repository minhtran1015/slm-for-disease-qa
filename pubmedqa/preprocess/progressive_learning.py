#!/usr/bin/env python3
"""
Progressive and Curriculum Learning for Imbalanced PubMedQA
Author: Generated for PubMedQA project
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import re

def calculate_sample_difficulty(pmid: str, data: Dict) -> float:
    """Calculate difficulty score for a sample"""
    
    contexts = data.get('CONTEXTS', [])
    question = data.get('QUESTION', '')
    meshes = data.get('MESHES', [])
    
    # Text complexity metrics
    total_text = ' '.join(contexts) + ' ' + question
    
    # Length-based difficulty
    text_length = len(total_text)
    avg_sentence_length = len(total_text) / max(len(contexts), 1)
    
    # Vocabulary complexity
    words = total_text.lower().split()
    unique_words = len(set(words))
    vocabulary_richness = unique_words / max(len(words), 1)
    
    # Medical complexity
    mesh_count = len(meshes)
    
    # Syntactic complexity (rough approximation)
    comma_count = total_text.count(',')
    semicolon_count = total_text.count(';')
    parentheses_count = total_text.count('(')
    
    # Negation complexity
    negation_patterns = ['not', 'no ', 'without', 'absence', 'lack', 'failed', 'negative', 'versus', 'compared']
    negation_count = sum(total_text.lower().count(pattern) for pattern in negation_patterns)
    
    # Statistical terms (indicate complexity)
    stats_terms = ['significant', 'correlation', 'regression', 'odds ratio', 'confidence interval', 
                   'p-value', 'statistical', 'analysis', 'hypothesis']
    stats_count = sum(total_text.lower().count(term) for term in stats_terms)
    
    # Numbers and measurements (complexity indicators)
    number_count = len(re.findall(r'\d+\.?\d*', total_text))
    
    # Combine features into difficulty score
    difficulty = (
        text_length * 0.0001 +           # Longer = harder
        avg_sentence_length * 0.01 +     # Longer sentences = harder  
        vocabulary_richness * 2 +        # Rich vocabulary = harder
        mesh_count * 0.1 +               # More MeSH terms = harder
        (comma_count + semicolon_count) * 0.05 +  # Complex syntax = harder
        parentheses_count * 0.1 +        # Parenthetical info = harder
        negation_count * 0.2 +           # Negations = harder
        stats_count * 0.3 +              # Statistics = harder
        number_count * 0.05              # Numbers = slightly harder
    )
    
    return difficulty

def create_curriculum_datasets():
    """Create curriculum learning datasets ordered by difficulty"""
    
    print("="*60)
    print("CREATING CURRICULUM LEARNING DATASETS")
    print("="*60)
    
    # Load data
    with open("../data/pqaa_train_splits/yes.json", 'r') as f:
        yes_data = json.load(f)
    with open("../data/pqaa_train_splits/no.json", 'r') as f:
        no_data = json.load(f)
    
    print("Calculating difficulty scores...")
    
    # Calculate difficulties
    yes_difficulties = []
    for pmid, data in yes_data.items():
        difficulty = calculate_sample_difficulty(pmid, data)
        yes_difficulties.append((pmid, data, difficulty))
    
    no_difficulties = []
    for pmid, data in no_data.items():
        difficulty = calculate_sample_difficulty(pmid, data)
        no_difficulties.append((pmid, data, difficulty))
    
    # Sort by difficulty
    yes_difficulties.sort(key=lambda x: x[2])
    no_difficulties.sort(key=lambda x: x[2])
    
    print(f"Difficulty range - YES: {yes_difficulties[0][2]:.2f} to {yes_difficulties[-1][2]:.2f}")
    print(f"Difficulty range - NO: {no_difficulties[0][2]:.2f} to {no_difficulties[-1][2]:.2f}")
    
    # Create curriculum stages
    num_stages = 5
    curriculum_datasets = []
    
    # Calculate stage sizes (progressive increase)
    total_samples = min(len(yes_difficulties), len(no_difficulties))
    
    for stage in range(num_stages):
        # Progressive curriculum: start with 20%, increase by 20% each stage
        stage_ratio = (stage + 1) * 0.2
        stage_size = int(total_samples * stage_ratio)
        
        print(f"Creating curriculum stage {stage + 1}/{num_stages} (size: {stage_size * 2:,})...")
        
        curriculum_data = {}
        
        # Take samples from easy to current stage difficulty
        for pmid, data, _ in yes_difficulties[:stage_size]:
            curriculum_data[pmid] = data
        
        for pmid, data, _ in no_difficulties[:stage_size]:
            curriculum_data[pmid] = data
        
        # Save stage dataset
        output_path = f"../data/curriculum_stage_{stage + 1}.json"
        with open(output_path, 'w') as f:
            json.dump(curriculum_data, f, indent=4)
        
        curriculum_datasets.append({
            'stage': stage + 1,
            'file': f'curriculum_stage_{stage + 1}.json',
            'samples_per_class': stage_size,
            'total_samples': stage_size * 2,
            'difficulty_range': f"Easy to {stage_ratio * 100:.0f}% percentile",
            'description': f'Curriculum stage {stage + 1}: easiest {stage_ratio * 100:.0f}% of samples'
        })
    
    return curriculum_datasets, yes_difficulties, no_difficulties

def create_anti_curriculum_datasets(yes_difficulties: List, no_difficulties: List):
    """Create anti-curriculum (hard first) datasets"""
    
    print("Creating anti-curriculum datasets (hard first)...")
    
    # Reverse the order - start with hardest
    yes_hard_first = list(reversed(yes_difficulties))
    no_hard_first = list(reversed(no_difficulties))
    
    num_stages = 3
    anti_curriculum_datasets = []
    total_samples = min(len(yes_hard_first), len(no_hard_first))
    
    for stage in range(num_stages):
        stage_ratio = (stage + 1) * 0.33  # 33%, 66%, 100%
        stage_size = int(total_samples * stage_ratio)
        
        print(f"Creating anti-curriculum stage {stage + 1}/{num_stages}...")
        
        curriculum_data = {}
        
        # Take samples from hardest to current stage
        for pmid, data, _ in yes_hard_first[:stage_size]:
            curriculum_data[pmid] = data
        
        for pmid, data, _ in no_hard_first[:stage_size]:
            curriculum_data[pmid] = data
        
        output_path = f"../data/anti_curriculum_stage_{stage + 1}.json"
        with open(output_path, 'w') as f:
            json.dump(curriculum_data, f, indent=4)
        
        anti_curriculum_datasets.append({
            'stage': stage + 1,
            'file': f'anti_curriculum_stage_{stage + 1}.json',
            'samples_per_class': stage_size,
            'total_samples': stage_size * 2,
            'description': f'Anti-curriculum stage {stage + 1}: hardest {stage_ratio * 100:.0f}% of samples'
        })
    
    return anti_curriculum_datasets

def create_mixed_curriculum_datasets(yes_difficulties: List, no_difficulties: List):
    """Create mixed difficulty curriculum datasets"""
    
    print("Creating mixed curriculum datasets...")
    
    mixed_datasets = []
    total_samples = min(len(yes_difficulties), len(no_difficulties))
    
    # Strategy 1: Sandwich curriculum (easy-hard-easy)
    stage_sizes = [
        int(total_samples * 0.3),  # Easy 30%
        int(total_samples * 0.5),  # Add hard 20% 
        int(total_samples * 0.8)   # Add medium 30%
    ]
    
    for stage_idx, stage_size in enumerate(stage_sizes):
        curriculum_data = {}
        
        if stage_idx == 0:  # Easy samples
            for pmid, data, _ in yes_difficulties[:stage_size]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[:stage_size]:
                curriculum_data[pmid] = data
        
        elif stage_idx == 1:  # Add hardest samples
            # Keep easy samples
            for pmid, data, _ in yes_difficulties[:stage_sizes[0]]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[:stage_sizes[0]]:
                curriculum_data[pmid] = data
            
            # Add hardest samples
            hard_count = stage_size - stage_sizes[0]
            for pmid, data, _ in yes_difficulties[-hard_count:]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[-hard_count:]:
                curriculum_data[pmid] = data
        
        else:  # Add medium difficulty samples
            # Keep previous samples
            for pmid, data, _ in yes_difficulties[:stage_sizes[0]]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[:stage_sizes[0]]:
                curriculum_data[pmid] = data
            
            hard_count = stage_sizes[1] - stage_sizes[0]
            for pmid, data, _ in yes_difficulties[-hard_count:]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[-hard_count:]:
                curriculum_data[pmid] = data
            
            # Add medium samples
            medium_start = stage_sizes[0]
            medium_end = total_samples - hard_count
            medium_count = stage_size - stage_sizes[1]
            
            for pmid, data, _ in yes_difficulties[medium_start:medium_start + medium_count]:
                curriculum_data[pmid] = data
            for pmid, data, _ in no_difficulties[medium_start:medium_start + medium_count]:
                curriculum_data[pmid] = data
        
        output_path = f"../data/sandwich_curriculum_stage_{stage_idx + 1}.json"
        with open(output_path, 'w') as f:
            json.dump(curriculum_data, f, indent=4)
        
        mixed_datasets.append({
            'stage': stage_idx + 1,
            'file': f'sandwich_curriculum_stage_{stage_idx + 1}.json',
            'total_samples': len(curriculum_data),
            'strategy': 'sandwich',
            'description': f'Sandwich curriculum stage {stage_idx + 1}'
        })
    
    return mixed_datasets

def create_progressive_learning_config():
    """Create configuration for progressive learning strategies"""
    
    config = {
        'curriculum_learning': {
            'description': 'Train on progressively harder examples',
            'advantages': ['Better convergence', 'Reduced overfitting', 'Faster initial learning'],
            'training_schedule': {
                'stage_1': {'epochs': 2, 'lr': 1e-4, 'description': 'Easiest 20% samples'},
                'stage_2': {'epochs': 2, 'lr': 8e-5, 'description': 'Easiest 40% samples'}, 
                'stage_3': {'epochs': 2, 'lr': 6e-5, 'description': 'Easiest 60% samples'},
                'stage_4': {'epochs': 2, 'lr': 4e-5, 'description': 'Easiest 80% samples'},
                'stage_5': {'epochs': 3, 'lr': 2e-5, 'description': 'All samples'}
            }
        },
        
        'anti_curriculum': {
            'description': 'Train on hardest examples first', 
            'advantages': ['Better generalization', 'Robust to noise', 'Good for imbalanced data'],
            'training_schedule': {
                'stage_1': {'epochs': 3, 'lr': 5e-5, 'description': 'Hardest 33% samples'},
                'stage_2': {'epochs': 2, 'lr': 3e-5, 'description': 'Hardest 66% samples'},
                'stage_3': {'epochs': 2, 'lr': 2e-5, 'description': 'All samples'}
            }
        },
        
        'sandwich_curriculum': {
            'description': 'Easy -> Hard -> Medium progression',
            'advantages': ['Combines benefits of both approaches', 'Stable learning'],
            'training_schedule': {
                'stage_1': {'epochs': 2, 'lr': 1e-4, 'description': 'Easy samples foundation'},
                'stage_2': {'epochs': 2, 'lr': 5e-5, 'description': 'Add hardest samples'},
                'stage_3': {'epochs': 3, 'lr': 2e-5, 'description': 'Fill with medium samples'}
            }
        },
        
        'implementation_tips': {
            'pytorch': 'Use different DataLoaders for each stage',
            'transformers': 'Reload trainer with new dataset each stage', 
            'evaluation': 'Validate on same test set after each stage',
            'early_stopping': 'Monitor validation loss, not training loss'
        }
    }
    
    return config

def main():
    """Main function to create all progressive learning datasets"""
    
    random.seed(42)
    np.random.seed(42)
    
    # Create curriculum datasets
    curriculum_datasets, yes_diff, no_diff = create_curriculum_datasets()
    
    # Create anti-curriculum datasets
    anti_curriculum_datasets = create_anti_curriculum_datasets(yes_diff, no_diff)
    
    # Create mixed curriculum datasets
    mixed_datasets = create_mixed_curriculum_datasets(yes_diff, no_diff)
    
    # Create configuration
    progressive_config = create_progressive_learning_config()
    
    # Combine all configurations
    complete_config = {
        'curriculum_datasets': curriculum_datasets,
        'anti_curriculum_datasets': anti_curriculum_datasets,
        'mixed_curriculum_datasets': mixed_datasets,
        'progressive_learning_config': progressive_config,
        'difficulty_analysis': {
            'yes_samples': len(yes_diff),
            'no_samples': len(no_diff),
            'difficulty_metrics': [
                'text_length', 'vocabulary_richness', 'mesh_complexity',
                'syntactic_complexity', 'negation_patterns', 'statistical_terms'
            ]
        }
    }
    
    # Save complete configuration
    output_path = "../data/progressive_learning_config.json"
    with open(output_path, 'w') as f:
        json.dump(complete_config, f, indent=4)
    
    print(f"\n" + "="*60)
    print("PROGRESSIVE LEARNING SETUP COMPLETE")
    print("="*60)
    
    print(f"Configuration saved to: {output_path}")
    
    print(f"\nCreated datasets:")
    print(f"📚 Curriculum Learning ({len(curriculum_datasets)} stages):")
    for ds in curriculum_datasets:
        print(f"  - {ds['file']}: {ds['description']}")
    
    print(f"\n🔥 Anti-Curriculum Learning ({len(anti_curriculum_datasets)} stages):")
    for ds in anti_curriculum_datasets:
        print(f"  - {ds['file']}: {ds['description']}")
    
    print(f"\n🥪 Sandwich Curriculum ({len(mixed_datasets)} stages):")
    for ds in mixed_datasets:
        print(f"  - {ds['file']}: {ds['description']}")
    
    print(f"\n🎯 Recommendations:")
    print(f"  1. Start with curriculum learning for stable training")
    print(f"  2. Try anti-curriculum for better generalization") 
    print(f"  3. Use sandwich curriculum for best of both worlds")
    print(f"  4. Always validate on the same test set across stages")

if __name__ == "__main__":
    main()