#!/usr/bin/env python3
"""
HPO to Vietnamese Symptom Relationship Dataset Converter

Converts Human Phenotype Ontology (HPO) to trainable Vietnamese medical QA format.
Generates yes/no questions about symptom relationships using graph traversal.

Target: 20,000 samples with balanced forward/reverse/hard-negative questions.
"""

import json
import random
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re

class HPOConverter:
    def __init__(self, hpo_json_path: str):
        self.hpo_json_path = hpo_json_path
        self.nodes = {}
        self.edges = []
        self.parent_child_map = defaultdict(list)  # parent -> children
        self.child_parent_map = defaultdict(list)  # child -> parents
        self.term_labels = {}  # id -> label

        self.load_hpo_data()

    def load_hpo_data(self):
        """Load HPO JSON data and build relationship maps."""
        print("Loading HPO data...")
        with open(self.hpo_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load nodes
        for node in data['graphs'][0]['nodes']:
            node_id = node['id']
            label = node.get('lbl', '')
            self.term_labels[node_id] = label
            self.nodes[node_id] = node

        # Load edges (relationships)
        for edge in data['graphs'][0]['edges']:
            if edge['pred'] == 'is_a':  # Only inheritance relationships
                parent = edge['obj']
                child = edge['sub']
                self.edges.append(edge)
                self.parent_child_map[parent].append(child)
                self.child_parent_map[child].append(parent)

        print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} is_a relationships")

    def get_term_label(self, term_id: str) -> str:
        """Get human-readable label for HPO term."""
        return self.term_labels.get(term_id, term_id)

    def simple_vietnamese_translation(self, english_text: str) -> str:
        """
        Simple translation mapping for common medical terms.
        In production, this should use a proper medical translation service.
        """
        # Basic medical term translations (expand as needed)
        translations = {
            # Cardiovascular
            "Heart failure": "Suy tim",
            "Cardiovascular abnormality": "Rối loạn tim mạch",
            "Hypertension": "Tăng huyết áp",
            "Arrhythmia": "Rối loạn nhịp tim",

            # Respiratory
            "Respiratory failure": "Suy hô hấp",
            "Pneumonia": "Viêm phổi",
            "Asthma": "Hen suyễn",

            # Neurological
            "Seizures": "Co giật",
            "Epilepsy": "Động kinh",
            "Headache": "Đau đầu",
            "Dementia": "Sa sút trí tuệ",

            # Gastrointestinal
            "Abdominal pain": "Đau bụng",
            "Nausea": "Buồn nôn",
            "Vomiting": "Nôn mửa",
            "Diarrhea": "Tiêu chảy",

            # Musculoskeletal
            "Arthritis": "Viêm khớp",
            "Joint pain": "Đau khớp",
            "Muscle weakness": "Yếu cơ",

            # Common patterns
            "Abnormality of": "Rối loạn",
            "Abnormal": "Bất thường",
            "Syndrome": "Hội chứng",
            "Disease": "Bệnh",
            "Disorder": "Rối loạn",

            # Generic terms
            "Pain": "Đau",
            "Fever": "Sốt",
            "Fatigue": "Mệt mỏi",
            "Anemia": "Thiếu máu",
        }

        result = english_text
        for eng, vie in translations.items():
            result = re.sub(r'\b' + re.escape(eng) + r'\b', vie, result, flags=re.IGNORECASE)

        return result

    def generate_relationship_samples(self, max_samples: int = 20000) -> List[Dict]:
        """
        Generate training samples from HPO relationships.

        Types of questions:
        1. Forward: "Is [child] a type of [parent]?" -> Yes
        2. Reverse: "Is [parent] a type of [child]?" -> No
        3. Hard negative: "Is [unrelated_term] a type of [parent]?" -> No
        """
        samples = []

        # Get all valid relationships (skip root term)
        root_term = "http://purl.obolibrary.org/obo/HP_0000001"
        valid_relationships = [
            (child, parent) for parent, children in self.parent_child_map.items()
            for child in children
            if parent != root_term and child != root_term
        ]

        print(f"Found {len(valid_relationships)} valid parent-child relationships")

        # Sample relationships to avoid generating too many
        sampled_relationships = random.sample(
            valid_relationships,
            min(len(valid_relationships), max_samples // 3)
        )

        for child_id, parent_id in sampled_relationships:
            child_label = self.get_term_label(child_id)
            parent_label = self.get_term_label(parent_id)

            if not child_label or not parent_label:
                continue

            # Vietnamese translations
            child_vn = self.simple_vietnamese_translation(child_label)
            parent_vn = self.simple_vietnamese_translation(parent_label)

            # 1. Forward question (Yes)
            forward_question = f"{child_vn} có phải là một dạng của {parent_vn} không?"
            samples.append({
                "instruction": "Dựa vào kiến thức về triệu chứng y khoa, hãy trả lời Đúng hoặc Sai.",
                "input": forward_question,
                "output": "Đúng",
                "question_type": "forward",
                "relationship": "child_to_parent",
                "hpo_child": child_label,
                "hpo_parent": parent_label,
                "hpo_child_vn": child_vn,
                "hpo_parent_vn": parent_vn
            })

            # 2. Reverse question (No)
            reverse_question = f"{parent_vn} có phải là một dạng của {child_vn} không?"
            samples.append({
                "instruction": "Dựa vào kiến thức về triệu chứng y khoa, hãy trả lời Đúng hoặc Sai.",
                "input": reverse_question,
                "output": "Sai",
                "question_type": "reverse",
                "relationship": "parent_to_child",
                "hpo_child": child_label,
                "hpo_parent": parent_label,
                "hpo_child_vn": child_vn,
                "hpo_parent_vn": parent_vn
            })

            # 3. Hard negative (No) - find unrelated term from same domain
            unrelated_term = self.find_unrelated_term(parent_id, child_id)
            if unrelated_term:
                unrelated_label = self.get_term_label(unrelated_term)
                unrelated_vn = self.simple_vietnamese_translation(unrelated_label)

                hard_neg_question = f"{unrelated_vn} có phải là một dạng của {parent_vn} không?"
                samples.append({
                    "instruction": "Dựa vào kiến thức về triệu chứng y khoa, hãy trả lời Đúng hoặc Sai.",
                    "input": hard_neg_question,
                    "output": "Sai",
                    "question_type": "hard_negative",
                    "relationship": "cross_category_false",
                    "hpo_child": unrelated_label,
                    "hpo_parent": parent_label,
                    "hpo_child_vn": unrelated_vn,
                    "hpo_parent_vn": parent_vn
                })

        # Shuffle and limit to target size
        random.shuffle(samples)
        return samples[:max_samples]

    def find_unrelated_term(self, parent_id: str, exclude_child: str) -> str:
        """Find a term unrelated to the parent but in similar domain."""
        # Simple approach: find terms that don't share common ancestors
        # In production, this could be more sophisticated

        parent_ancestors = self.get_ancestors(parent_id)
        parent_ancestors.add(parent_id)

        # Get all terms not in parent's ancestry
        candidates = []
        for term_id in self.term_labels.keys():
            if term_id not in parent_ancestors and term_id != exclude_child:
                term_ancestors = self.get_ancestors(term_id)
                # Avoid terms that are too closely related
                if not parent_ancestors.intersection(term_ancestors):
                    candidates.append(term_id)

        return random.choice(candidates) if candidates else None

    def get_ancestors(self, term_id: str) -> Set[str]:
        """Get all ancestors of a term."""
        ancestors = set()
        to_visit = [term_id]

        while to_visit:
            current = to_visit.pop()
            for parent in self.child_parent_map.get(current, []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    to_visit.append(parent)

        return ancestors

def main():
    converter = HPOConverter("hp.json")

    print("Generating HPO relationship samples...")
    samples = converter.generate_relationship_samples(max_samples=20000)

    print(f"Generated {len(samples)} samples")

    # Split into train/test (90/10)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    # Save to files
    with open("hpo_vietnamese_symptoms_train.jsonl", 'w', encoding='utf-8') as f:
        for sample in train_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    with open("hpo_vietnamese_symptoms_test.jsonl", 'w', encoding='utf-8') as f:
        for sample in test_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    # Create analysis file
    analysis = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "question_types": {
            "forward": len([s for s in samples if s["question_type"] == "forward"]),
            "reverse": len([s for s in samples if s["question_type"] == "reverse"]),
            "hard_negative": len([s for s in samples if s["question_type"] == "hard_negative"])
        },
        "note": "Vietnamese translations are basic mappings. For production, use professional medical translation service."
    }

    with open("hpo_vietnamese_symptoms_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("Files created:")
    print(f"- hpo_vietnamese_symptoms_train.jsonl: {len(train_samples)} samples")
    print(f"- hpo_vietnamese_symptoms_test.jsonl: {len(test_samples)} samples")
    print(f"- hpo_vietnamese_symptoms_analysis.json: analysis data")

if __name__ == "__main__":
    main()