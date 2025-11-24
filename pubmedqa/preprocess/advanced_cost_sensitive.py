#!/usr/bin/env python3
"""
Advanced cost-sensitive learning strategies for imbalanced PubMedQA
Author: Generated for PubMedQA project
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import math

def calculate_advanced_costs():
    """Calculate various cost matrices for cost-sensitive learning"""
    
    # Load data to get true distribution
    with open("../data/pqaa_train_splits/yes.json", 'r') as f:
        yes_data = json.load(f)
    with open("../data/pqaa_train_splits/no.json", 'r') as f:
        no_data = json.load(f)
    
    n_yes = len(yes_data)
    n_no = len(no_data)
    total = n_yes + n_no
    
    # Calculate various cost matrices
    costs = {}
    
    # 1. Inverse Frequency Weighting
    costs['inverse_frequency'] = {
        'yes_cost': total / (2 * n_yes),
        'no_cost': total / (2 * n_no),
        'description': 'Standard inverse frequency weighting'
    }
    
    # 2. Balanced Class Weights (scikit-learn style)
    costs['balanced'] = {
        'yes_cost': total / (2 * n_yes),
        'no_cost': total / (2 * n_no),
        'description': 'Balanced class weights'
    }
    
    # 3. Log-based Weighting (for extreme imbalance)
    yes_weight = math.log(total / n_yes)
    no_weight = math.log(total / n_no)
    total_log_weight = yes_weight + no_weight
    
    costs['log_weighted'] = {
        'yes_cost': yes_weight / total_log_weight,
        'no_cost': no_weight / total_log_weight,
        'description': 'Log-based weighting for extreme imbalance'
    }
    
    # 4. Effective Number Based (from CB-Focal Loss paper)
    beta = 0.9999  # Hyperparameter
    effective_yes = (1 - beta**n_yes) / (1 - beta)
    effective_no = (1 - beta**n_no) / (1 - beta)
    
    costs['effective_number'] = {
        'yes_cost': effective_no / (effective_yes + effective_no),
        'no_cost': effective_yes / (effective_yes + effective_no),
        'description': 'Effective number based weighting (CB-Focal Loss)'
    }
    
    # 5. Sqrt Weighting (gentler than inverse)
    sqrt_yes = math.sqrt(n_yes)
    sqrt_no = math.sqrt(n_no)
    sqrt_total = sqrt_yes + sqrt_no
    
    costs['sqrt_weighted'] = {
        'yes_cost': sqrt_no / sqrt_total,
        'no_cost': sqrt_yes / sqrt_total,
        'description': 'Square root based weighting (gentler)'
    }
    
    # 6. Custom Medical Domain Weights
    # In medical domain, false negatives might be more costly
    fn_penalty = 10.0  # False negative penalty multiplier
    fp_penalty = 1.0   # False positive penalty
    
    costs['medical_domain'] = {
        'yes_cost': fp_penalty,
        'no_cost': fn_penalty * (n_yes / n_no),  # Scale by imbalance
        'description': 'Medical domain specific (FN penalty = 10x FP)'
    }
    
    # Add statistics
    costs['statistics'] = {
        'n_yes': n_yes,
        'n_no': n_no,
        'imbalance_ratio': n_yes / n_no,
        'minority_percentage': (n_no / total) * 100
    }
    
    return costs

def generate_cost_matrices():
    """Generate cost matrices for different scenarios"""
    
    cost_matrices = {}
    
    # Standard cost matrix (equal costs)
    cost_matrices['equal'] = {
        'matrix': [[0, 1], [1, 0]],  # [TN, FP], [FN, TP]
        'description': 'Equal cost for all errors'
    }
    
    # Medical domain: FN more costly
    cost_matrices['medical_conservative'] = {
        'matrix': [[0, 1], [10, 0]],  # FN cost = 10x FP cost
        'description': 'Conservative medical: False negatives 10x more costly'
    }
    
    # Screening scenario: FP slightly more costly
    cost_matrices['screening'] = {
        'matrix': [[0, 3], [5, 0]], # Balance between FN and FP
        'description': 'Screening scenario: Moderate FN penalty'
    }
    
    # Research scenario: Balanced but weighted
    cost_matrices['research'] = {
        'matrix': [[0, 2], [3, 0]],
        'description': 'Research scenario: Slight FN penalty'
    }
    
    return cost_matrices

def focal_loss_variants():
    """Generate parameters for different focal loss variants"""
    
    variants = {}
    
    # Standard focal loss
    variants['standard'] = {
        'alpha': 0.25,  # Standard value
        'gamma': 2.0,   # Standard focusing parameter
        'description': 'Standard focal loss parameters'
    }
    
    # High imbalance focal loss
    variants['high_imbalance'] = {
        'alpha': 0.072,  # Based on minority class percentage
        'gamma': 3.0,    # Stronger focusing
        'description': 'Tuned for high imbalance (13:1 ratio)'
    }
    
    # Gentle focal loss
    variants['gentle'] = {
        'alpha': 0.2,
        'gamma': 1.0,   # Less aggressive focusing
        'description': 'Gentler focusing for stable training'
    }
    
    # Adaptive focal loss
    variants['adaptive'] = {
        'alpha': 0.072,
        'gamma': 2.5,
        'label_smoothing': 0.1,  # Add label smoothing
        'description': 'Adaptive with label smoothing'
    }
    
    return variants

def threshold_strategies():
    """Generate optimal threshold strategies"""
    
    strategies = {
        'default': {
            'threshold': 0.5,
            'description': 'Default 0.5 threshold'
        },
        'precision_focused': {
            'threshold': 0.7,
            'description': 'Higher threshold for better precision'
        },
        'recall_focused': {
            'threshold': 0.3,
            'description': 'Lower threshold for better recall'
        },
        'f1_optimized': {
            'threshold': 'optimize_f1',
            'description': 'Use validation set to find F1-optimal threshold'
        },
        'cost_optimized': {
            'threshold': 'optimize_cost',
            'description': 'Use validation set to find cost-optimal threshold'
        }
    }
    
    return strategies

def main():
    """Generate comprehensive cost-sensitive learning configuration"""
    
    print("="*60)
    print("ADVANCED COST-SENSITIVE LEARNING CONFIGURATION")
    print("="*60)
    
    # Generate all configurations
    print("Calculating advanced cost strategies...")
    costs = calculate_advanced_costs()
    
    print("Generating cost matrices...")
    cost_matrices = generate_cost_matrices()
    
    print("Setting up focal loss variants...")
    focal_variants = focal_loss_variants()
    
    print("Defining threshold strategies...")
    threshold_strats = threshold_strategies()
    
    # Combine into comprehensive config
    advanced_config = {
        'cost_weights': costs,
        'cost_matrices': cost_matrices,
        'focal_loss_variants': focal_variants,
        'threshold_strategies': threshold_strats,
        'recommended_pipeline': {
            'phase_1': {
                'strategy': 'effective_number',
                'focal_variant': 'high_imbalance',
                'description': 'Start with effective number weighting + high imbalance focal loss'
            },
            'phase_2': {
                'strategy': 'medical_domain',
                'focal_variant': 'adaptive',
                'description': 'Fine-tune with domain-specific costs'
            },
            'phase_3': {
                'threshold_strategy': 'f1_optimized',
                'description': 'Optimize threshold on validation set'
            }
        },
        'implementation_notes': {
            'pytorch': 'Use nn.CrossEntropyLoss(weight=torch.tensor([no_cost, yes_cost]))',
            'tensorflow': 'Use class_weight parameter in model.fit()',
            'transformers': 'Override compute_loss in custom Trainer class',
            'threshold_tuning': 'Use sklearn.metrics.precision_recall_curve for threshold optimization'
        }
    }
    
    # Save configuration
    output_path = "../data/advanced_cost_sensitive_config.json"
    with open(output_path, 'w') as f:
        json.dump(advanced_config, f, indent=4)
    
    print(f"\nAdvanced configuration saved to: {output_path}")
    
    # Print summary
    print(f"\nKey Recommendations:")
    print(f"1. Start with 'effective_number' weighting:")
    print(f"   - YES cost: {costs['effective_number']['yes_cost']:.4f}")
    print(f"   - NO cost: {costs['effective_number']['no_cost']:.4f}")
    
    print(f"\n2. Use high-imbalance focal loss:")
    print(f"   - Alpha: {focal_variants['high_imbalance']['alpha']}")
    print(f"   - Gamma: {focal_variants['high_imbalance']['gamma']}")
    
    print(f"\n3. Consider medical domain costs (FN penalty):")
    print(f"   - YES cost: {costs['medical_domain']['yes_cost']:.4f}")
    print(f"   - NO cost: {costs['medical_domain']['no_cost']:.4f}")
    
    print(f"\n4. Optimize threshold on validation set for F1-score")

if __name__ == "__main__":
    main()