#!/usr/bin/env python3
"""
SMOTE-based data augmentation for PubMedQA
Creates synthetic minority class samples using semantic similarity
Author: Generated for PubMedQA project
"""

import json
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import re

def extract_features(contexts: List[str]) -> Dict:
    """Extract simple features from contexts for SMOTE"""
    
    # Combine all contexts
    text = " ".join(contexts).lower()
    
    # Basic features
    features = {
        'length': len(text),
        'sentence_count': len(contexts),
        'avg_sentence_length': len(text) / max(len(contexts), 1),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'numbers': len(re.findall(r'\d+', text)),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }
    
    # Medical/scientific keywords
    medical_keywords = [
        'patient', 'treatment', 'therapy', 'clinical', 'study', 'trial',
        'disease', 'symptom', 'diagnosis', 'outcome', 'efficacy', 'safety',
        'significant', 'correlation', 'association', 'risk', 'benefit'
    ]
    
    for keyword in medical_keywords:
        features[f'has_{keyword}'] = 1 if keyword in text else 0
        features[f'count_{keyword}'] = text.count(keyword)
    
    # Negation patterns
    negation_patterns = ['not', 'no ', 'without', 'absence', 'lack', 'failed', 'negative']
    features['negation_count'] = sum(text.count(pattern) for pattern in negation_patterns)
    
    return features

def semantic_smote_augmentation(minority_samples: List[Tuple], k: int = 5, augment_factor: int = 2):
    """
    Create synthetic samples using semantic features
    
    Args:
        minority_samples: List of (pmid, data) tuples for minority class
        k: Number of nearest neighbors to consider
        augment_factor: How many synthetic samples per original sample
    """
    
    print(f"Extracting features from {len(minority_samples)} minority samples...")
    
    # Extract features for all samples
    features_list = []
    for pmid, data in minority_samples:
        features = extract_features(data['CONTEXTS'])
        features_list.append((pmid, data, features))
    
    synthetic_samples = {}
    synthetic_count = 0
    
    print(f"Generating synthetic samples (factor: {augment_factor})...")
    
    for i, (pmid, data, features) in enumerate(features_list):
        # Find k nearest neighbors based on feature similarity
        similarities = []
        
        for j, (other_pmid, other_data, other_features) in enumerate(features_list):
            if i != j:
                # Calculate simple euclidean distance on normalized features
                feature_keys = set(features.keys()) & set(other_features.keys())
                
                if feature_keys:
                    diff_sum = sum((features[key] - other_features[key]) ** 2 
                                 for key in feature_keys)
                    distance = np.sqrt(diff_sum / len(feature_keys))
                    similarities.append((distance, j))
        
        # Get k nearest neighbors
        similarities.sort()
        nearest_neighbors = similarities[:min(k, len(similarities))]
        
        # Generate synthetic samples
        for aug_idx in range(augment_factor):
            if nearest_neighbors:
                # Select random neighbor
                _, neighbor_idx = random.choice(nearest_neighbors)
                neighbor_pmid, neighbor_data, neighbor_features = features_list[neighbor_idx]
                
                # Create synthetic sample by interpolating contexts
                synthetic_data = data.copy()
                
                # Interpolate contexts (simple approach: mix sentences)
                original_contexts = data['CONTEXTS']
                neighbor_contexts = neighbor_data['CONTEXTS']
                
                # Create mixed contexts
                mixed_contexts = []
                max_contexts = max(len(original_contexts), len(neighbor_contexts))
                
                for ctx_idx in range(max_contexts):
                    if random.random() < 0.6:  # Favor original
                        if ctx_idx < len(original_contexts):
                            mixed_contexts.append(original_contexts[ctx_idx])
                        elif neighbor_contexts:
                            mixed_contexts.append(random.choice(neighbor_contexts))
                    else:  # Use neighbor
                        if ctx_idx < len(neighbor_contexts):
                            mixed_contexts.append(neighbor_contexts[ctx_idx])
                        elif original_contexts:
                            mixed_contexts.append(random.choice(original_contexts))
                
                synthetic_data['CONTEXTS'] = mixed_contexts
                
                # Mix other fields
                synthetic_data['MESHES'] = list(set(data.get('MESHES', []) + 
                                                  neighbor_data.get('MESHES', [])))
                synthetic_data['LABELS'] = data.get('LABELS', [])  # Keep original labels
                
                # Create synthetic PMID
                synthetic_pmid = f"SMOTE_{pmid}_{neighbor_pmid}_{aug_idx}"
                synthetic_samples[synthetic_pmid] = synthetic_data
                synthetic_count += 1
    
    print(f"Generated {synthetic_count} synthetic samples")
    return synthetic_samples

def augment_minority_class():
    """Augment the minority class using SMOTE-like approach"""
    
    print("="*60)
    print("SMOTE-BASED MINORITY CLASS AUGMENTATION")
    print("="*60)
    
    # Load minority class data
    with open("../data/pqaa_train_splits/no.json", 'r') as f:
        no_data = json.load(f)
    
    minority_samples = [(pmid, data) for pmid, data in no_data.items()]
    
    print(f"Original minority class size: {len(minority_samples)}")
    
    # Generate synthetic samples
    synthetic_samples = semantic_smote_augmentation(
        minority_samples, 
        k=5, 
        augment_factor=3  # Triple the minority class
    )
    
    # Combine original and synthetic
    augmented_data = no_data.copy()
    augmented_data.update(synthetic_samples)
    
    print(f"Augmented minority class size: {len(augmented_data)}")
    
    # Save augmented dataset
    output_path = "../data/pqaa_no_augmented_smote.json"
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    
    print(f"Augmented dataset saved to: {output_path}")
    
    return len(synthetic_samples)

def create_smote_balanced_dataset():
    """Create fully balanced dataset using SMOTE augmentation"""
    
    print("\n" + "="*60)
    print("CREATING SMOTE-BALANCED DATASET")
    print("="*60)
    
    # Load majority and augmented minority
    with open("../data/pqaa_train_splits/yes.json", 'r') as f:
        yes_data = json.load(f)
    
    with open("../data/pqaa_no_augmented_smote.json", 'r') as f:
        no_augmented = json.load(f)
    
    # Calculate target size (geometric mean for balance)
    target_size = int(np.sqrt(len(yes_data) * len(no_augmented)))
    
    print(f"YES samples: {len(yes_data):,}")
    print(f"NO samples (augmented): {len(no_augmented):,}")
    print(f"Target balanced size per class: {target_size:,}")
    
    # Sample from both classes
    yes_pmids = list(yes_data.keys())
    no_pmids = list(no_augmented.keys())
    
    random.shuffle(yes_pmids)
    random.shuffle(no_pmids)
    
    balanced_data = {}
    
    # Add sampled YES
    for pmid in yes_pmids[:target_size]:
        balanced_data[pmid] = yes_data[pmid]
    
    # Add sampled NO
    for pmid in no_pmids[:target_size]:
        balanced_data[pmid] = no_augmented[pmid]
    
    print(f"Final balanced dataset size: {len(balanced_data):,}")
    
    # Save balanced dataset
    output_path = "../data/pqaa_train_balanced_smote.json"
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=4)
    
    print(f"SMOTE-balanced dataset saved to: {output_path}")

def main():
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Augment minority class
    synthetic_count = augment_minority_class()
    
    # Step 2: Create balanced dataset
    create_smote_balanced_dataset()
    
    print(f"\n" + "="*60)
    print("SMOTE AUGMENTATION COMPLETE")
    print("="*60)
    print(f"Generated {synthetic_count} synthetic samples")
    print("Files created:")
    print("  - pqaa_no_augmented_smote.json (augmented minority class)")
    print("  - pqaa_train_balanced_smote.json (balanced with SMOTE)")

if __name__ == "__main__":
    main()