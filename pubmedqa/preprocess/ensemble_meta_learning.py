#!/usr/bin/env python3
"""
Ensemble and meta-learning strategies for imbalanced PubMedQA
Author: Generated for PubMedQA project
"""

import json
import random
import math
from typing import List, Dict, Tuple
from collections import defaultdict

def create_ensemble_datasets():
    """Create multiple diverse training sets for ensemble learning"""
    
    print("="*60)
    print("CREATING ENSEMBLE DATASETS")
    print("="*60)
    
    # Load data
    with open("../data/pqaa_train_splits/yes.json", 'r') as f:
        yes_data = json.load(f)
    with open("../data/pqaa_train_splits/no.json", 'r') as f:
        no_data = json.load(f)
    
    yes_pmids = list(yes_data.keys())
    no_pmids = list(no_data.keys())
    
    ensemble_configs = []
    
    # 1. Balanced Bootstrap Ensembles
    for i in range(5):
        print(f"Creating balanced bootstrap ensemble {i+1}/5...")
        
        # Sample with replacement from minority class
        sampled_no = random.choices(no_pmids, k=len(yes_pmids))
        # Sample without replacement from majority class
        sampled_yes = random.sample(yes_pmids, len(no_pmids))
        
        ensemble_data = {}
        
        # Add sampled data
        for pmid in sampled_no:
            ensemble_data[f"{pmid}_ens{i}"] = no_data[pmid].copy()
        
        for pmid in sampled_yes:
            ensemble_data[pmid] = yes_data[pmid].copy()
        
        output_path = f"../data/ensemble_balanced_{i+1}.json"
        with open(output_path, 'w') as f:
            json.dump(ensemble_data, f, indent=4)
        
        ensemble_configs.append({
            'file': f'ensemble_balanced_{i+1}.json',
            'strategy': 'balanced_bootstrap',
            'yes_samples': len(sampled_yes),
            'no_samples': len(sampled_no),
            'description': f'Balanced bootstrap ensemble {i+1}'
        })
    
    # 2. Easy/Hard Split Ensembles
    print("Creating difficulty-based ensembles...")
    
    # Simple heuristic for "difficulty": length and complexity
    def calculate_difficulty(data):
        contexts = data.get('CONTEXTS', [])
        text_length = sum(len(ctx) for ctx in contexts)
        avg_sentence_length = text_length / max(len(contexts), 1)
        mesh_count = len(data.get('MESHES', []))
        
        # Longer texts with more MeSH terms = harder
        return text_length * 0.4 + avg_sentence_length * 0.3 + mesh_count * 0.3
    
    # Calculate difficulties
    yes_difficulties = [(pmid, calculate_difficulty(data)) for pmid, data in yes_data.items()]
    no_difficulties = [(pmid, calculate_difficulty(data)) for pmid, data in no_data.items()]
    
    # Sort by difficulty
    yes_difficulties.sort(key=lambda x: x[1])
    no_difficulties.sort(key=lambda x: x[1])
    
    # Create easy/hard splits
    yes_mid = len(yes_difficulties) // 2
    no_mid = len(no_difficulties) // 2
    
    splits = [
        ('easy', yes_difficulties[:yes_mid], no_difficulties[:no_mid]),
        ('hard', yes_difficulties[yes_mid:], no_difficulties[no_mid:])
    ]
    
    for split_name, yes_subset, no_subset in splits:
        # Balance the subsets
        min_size = min(len(yes_subset), len(no_subset))
        
        ensemble_data = {}
        
        # Take equal amounts from each class
        for pmid, _ in yes_subset[:min_size]:
            ensemble_data[pmid] = yes_data[pmid]
        
        for pmid, _ in no_subset[:min_size]:
            ensemble_data[pmid] = no_data[pmid]
        
        output_path = f"../data/ensemble_{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(ensemble_data, f, indent=4)
        
        ensemble_configs.append({
            'file': f'ensemble_{split_name}.json',
            'strategy': f'{split_name}_samples',
            'yes_samples': min_size,
            'no_samples': min_size,
            'description': f'Ensemble trained on {split_name} samples'
        })
    
    # 3. Temporal Split Ensembles (by publication year)
    print("Creating temporal ensembles...")
    
    # Group by year
    yes_by_year = defaultdict(list)
    no_by_year = defaultdict(list)
    
    for pmid, data in yes_data.items():
        year = data.get('YEAR', '2000')
        yes_by_year[year].append((pmid, data))
    
    for pmid, data in no_data.items():
        year = data.get('YEAR', '2000')
        no_by_year[year].append((pmid, data))
    
    # Get available years
    all_years = sorted(set(yes_by_year.keys()) | set(no_by_year.keys()))
    
    if len(all_years) >= 3:
        # Create 3 temporal splits
        year_chunks = [
            all_years[:len(all_years)//3],
            all_years[len(all_years)//3:2*len(all_years)//3],
            all_years[2*len(all_years)//3:]
        ]
        
        for i, years in enumerate(year_chunks):
            ensemble_data = {}
            yes_count = 0
            no_count = 0
            
            for year in years:
                # Add all samples from this year range
                for pmid, data in yes_by_year.get(year, []):
                    ensemble_data[pmid] = data
                    yes_count += 1
                
                for pmid, data in no_by_year.get(year, []):
                    ensemble_data[pmid] = data
                    no_count += 1
            
            # Balance if needed
            if yes_count > 0 and no_count > 0:
                output_path = f"../data/ensemble_temporal_{i+1}.json"
                with open(output_path, 'w') as f:
                    json.dump(ensemble_data, f, indent=4)
                
                ensemble_configs.append({
                    'file': f'ensemble_temporal_{i+1}.json',
                    'strategy': f'temporal_{i+1}',
                    'yes_samples': yes_count,
                    'no_samples': no_count,
                    'years': years,
                    'description': f'Temporal ensemble {i+1} ({min(years)}-{max(years)})'
                })
    
    return ensemble_configs

def create_meta_learning_config():
    """Create configuration for meta-learning approaches"""
    
    meta_config = {
        'cascade_learning': {
            'description': 'Train models in cascade: easy -> hard examples',
            'stages': [
                {
                    'stage': 1,
                    'dataset': 'ensemble_easy.json',
                    'epochs': 10,
                    'learning_rate': 1e-4,
                    'description': 'Train on easy examples first'
                },
                {
                    'stage': 2,
                    'dataset': 'ensemble_hard.json',
                    'epochs': 5,
                    'learning_rate': 1e-5,
                    'description': 'Fine-tune on hard examples'
                }
            ]
        },
        
        'self_training': {
            'description': 'Semi-supervised learning with pseudo-labels',
            'steps': [
                'Train initial model on labeled data',
                'Predict on unlabeled PQA-U data',
                'Select high-confidence predictions',
                'Add pseudo-labels to training set',
                'Retrain model'
            ],
            'confidence_threshold': 0.9,
            'pseudo_label_ratio': 0.1
        },
        
        'co_training': {
            'description': 'Train multiple models on different views',
            'views': [
                {
                    'name': 'question_focused',
                    'features': ['QUESTION', 'first_context'],
                    'description': 'Focus on question and first context'
                },
                {
                    'name': 'context_focused', 
                    'features': ['CONTEXTS', 'MESHES'],
                    'description': 'Focus on all contexts and MeSH terms'
                }
            ],
            'agreement_threshold': 0.8
        },
        
        'active_learning': {
            'description': 'Iteratively select most informative samples',
            'strategies': [
                'uncertainty_sampling',
                'diversity_sampling', 
                'expected_model_change',
                'query_by_committee'
            ],
            'batch_size': 100,
            'iterations': 10
        }
    }
    
    return meta_config

def create_stacking_ensemble_config():
    """Create configuration for stacking ensemble"""
    
    stacking_config = {
        'base_models': [
            {
                'name': 'bert_balanced',
                'dataset': 'pqaa_train_balanced_undersample.json',
                'model_type': 'BERT',
                'hyperparams': {'lr': 2e-5, 'epochs': 3}
            },
            {
                'name': 'roberta_oversampled',
                'dataset': 'pqaa_train_balanced_oversample.json', 
                'model_type': 'RoBERTa',
                'hyperparams': {'lr': 1e-5, 'epochs': 5}
            },
            {
                'name': 'distilbert_smote',
                'dataset': 'pqaa_train_balanced_smote.json',
                'model_type': 'DistilBERT', 
                'hyperparams': {'lr': 3e-5, 'epochs': 4}
            },
            {
                'name': 'biobert_weighted',
                'dataset': 'pqaa_train_set.json',
                'model_type': 'BioBERT',
                'hyperparams': {'lr': 2e-5, 'epochs': 3},
                'use_class_weights': True
            }
        ],
        
        'meta_learner': {
            'model_type': 'LogisticRegression',
            'cross_validation': 5,
            'regularization': 'l2'
        },
        
        'ensemble_methods': [
            'simple_voting',
            'weighted_voting',
            'stacking',
            'blending'
        ],
        
        'validation_strategy': 'stratified_k_fold',
        'k_folds': 5
    }
    
    return stacking_config

def main():
    """Main function to create all ensemble configurations"""
    
    random.seed(42)
    
    # Create ensemble datasets
    ensemble_configs = create_ensemble_datasets()
    
    # Create meta-learning config
    meta_config = create_meta_learning_config()
    
    # Create stacking config
    stacking_config = create_stacking_ensemble_config()
    
    # Combine all configurations
    complete_config = {
        'ensemble_datasets': ensemble_configs,
        'meta_learning': meta_config,
        'stacking_ensemble': stacking_config,
        'recommendations': {
            'quick_ensemble': 'Train 3 models: balanced_undersample, balanced_oversample, original+weights',
            'advanced_ensemble': 'Use all 5 bootstrap ensembles + temporal ensembles',
            'meta_learning': 'Start with cascade learning (easy->hard)',
            'production': 'Use stacking ensemble with 4 diverse base models'
        }
    }
    
    # Save complete configuration
    output_path = "../data/ensemble_meta_learning_config.json"
    with open(output_path, 'w') as f:
        json.dump(complete_config, f, indent=4)
    
    print(f"\nEnsemble and meta-learning configuration saved to: {output_path}")
    print(f"\nCreated {len(ensemble_configs)} ensemble datasets:")
    
    for config in ensemble_configs:
        print(f"  - {config['file']}: {config['description']}")
    
    print(f"\nRecommended workflow:")
    print(f"1. Quick start: Train on 3 different balanced datasets")
    print(f"2. Advanced: Use all bootstrap ensembles") 
    print(f"3. Meta-learning: Try cascade learning (easy -> hard)")
    print(f"4. Production: Implement stacking ensemble")

if __name__ == "__main__":
    main()