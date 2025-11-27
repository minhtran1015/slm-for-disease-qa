#!/usr/bin/env python3
"""
HPO to Bilingual Vietnamese/English Symptom Dataset Converter

Uses Modal for GPU-accelerated translation with NLLB-200.
Generates yes/no questions with bilingual terms: Vietnamese (English)

Target: 20,000 samples (10,000 True + 10,000 False pairs)
"""

import modal
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ================= CONFIGURATION =================
SEED = 42
TARGET_PAIRS = 10000  # Will generate 20,000 total (10k True + 10k False)
BATCH_SIZE = 128  # Translation batch size (optimized for A10G)
TEST_MODE = False  # Set True to test with 1 batch only
TEST_BATCH_SIZE = 50  # Number of relationships for test mode
OUTPUT_TRAIN = "hpo_vietnamese_bilingual_train.jsonl"
OUTPUT_TEST = "hpo_vietnamese_bilingual_test.jsonl"
OUTPUT_STATS = "hpo_vietnamese_bilingual_stats.json"

# Modal configuration
app = modal.App("hpo-bilingual-translation")
volume = modal.Volume.from_name("medical-data", create_if_missing=True)

# GPU image with translation model
translation_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentencepiece",
        "accelerate",
        "tqdm"
    )
)

# Bilingual templates - Vietnamese question with English in parentheses
TEMPLATES_TRUE = [
    "{child_vi} ({child_en}) có phải là một dạng của {parent_vi} ({parent_en}) không?",
    "Trong y học, {child_vi} ({child_en}) thuộc nhóm {parent_vi} ({parent_en}) đúng không?",
    "Xác nhận: {child_vi} (hay {child_en}) là biểu hiện liên quan đến {parent_vi} ({parent_en})?",
    "{child_vi} ({child_en}) có thuộc loại {parent_vi} ({parent_en}) không?",
    "Triệu chứng {child_vi} ({child_en}) có nằm trong nhóm {parent_vi} ({parent_en}) không?",
    "{child_vi} ({child_en}) có liên quan đến {parent_vi} ({parent_en}) không?",
    "Biểu hiện {child_vi} ({child_en}) có phải là một phần của {parent_vi} ({parent_en}) không?",
    "{child_vi} ({child_en}) được xếp vào loại {parent_vi} ({parent_en}) phải không?",
]

TEMPLATES_FALSE = [
    "{child_vi} ({child_en}) có phải là một dạng của {fake_parent_vi} ({fake_parent_en}) không?",
    "Có phải {child_vi} ({child_en}) thuộc nhóm triệu chứng {fake_parent_vi} ({fake_parent_en})?",
    "Xác nhận: {child_vi} (hay {child_en}) là biểu hiện của {fake_parent_vi} ({fake_parent_en})?",
    "{child_vi} ({child_en}) có thuộc loại {fake_parent_vi} ({fake_parent_en}) không?",
    "Triệu chứng {child_vi} ({child_en}) có nằm trong nhóm {fake_parent_vi} ({fake_parent_en}) không?",
    "{child_vi} ({child_en}) có liên quan đến {fake_parent_vi} ({fake_parent_en}) không?",
    "Biểu hiện {child_vi} ({child_en}) có phải là một phần của {fake_parent_vi} ({fake_parent_en}) không?",
    "{child_vi} ({child_en}) được xếp vào loại {fake_parent_vi} ({fake_parent_en}) phải không?",
]

# Multiple instruction templates for variety
INSTRUCTIONS = [
    "Dựa trên kiến thức triệu chứng y học, trả lời Đúng hoặc Sai.",
    "Hãy cho biết câu sau đúng hay sai dựa vào kiến thức y khoa.",
    "Xác định tính đúng sai của nhận định sau về triệu chứng y học.",
    "Trả lời Đúng hoặc Sai cho câu hỏi y khoa sau.",
    "Dựa vào phân loại triệu chứng y học, hãy trả lời Đúng hoặc Sai.",
    "Với kiến thức về bệnh học, hãy xác nhận câu sau Đúng hay Sai.",
    "Câu hỏi về mối quan hệ triệu chứng - Trả lời Đúng hoặc Sai.",
    "Theo hệ thống phân loại y khoa, câu sau Đúng hay Sai?",
]


@dataclass
class HPORelationship:
    """Represents an HPO is_a relationship"""
    child_id: str
    child_en: str
    parent_id: str
    parent_en: str


@app.cls(
    image=translation_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/data": volume},
    retries=3
)
class BilingualTranslator:
    """GPU-accelerated translator using NLLB-200"""
    
    @modal.enter()
    def setup(self):
        """Load model on container start"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("🔄 Loading NLLB-200 translation model...")
        model_name = "facebook/nllb-200-distilled-600M"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).cuda()
        self.model.eval()
        
        # Source and target language codes for NLLB
        self.src_lang = "eng_Latn"
        self.tgt_lang = "vie_Latn"
        
        print("✅ Model loaded successfully!")
    
    def _translate_batch_internal(self, texts: List[str]) -> List[str]:
        """Internal method to translate a batch of English texts to Vietnamese"""
        import torch
        
        if not texts:
            return []
        
        # Clean texts (remove "obsolete" prefix common in HPO)
        cleaned_texts = [t.replace("obsolete ", "").strip() for t in texts]
        
        # Set source language
        self.tokenizer.src_lang = self.src_lang
        
        # Tokenize batch
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to("cuda")
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        # Fallback: if translation is same as input, keep English
        results = []
        for orig, trans in zip(cleaned_texts, translations):
            if trans.lower() == orig.lower() or not trans.strip():
                results.append(orig)
            else:
                results.append(trans)
        
        return results
    
    @modal.method()
    def translate_relationships(
        self, 
        relationships: List[Dict],
        all_node_labels: List[str]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Translate HPO relationships and generate bilingual QA pairs.
        
        Args:
            relationships: List of dicts with child_id, child_en, parent_id, parent_en
            all_node_labels: List of all HPO term labels for fake parent selection
            
        Returns:
            Tuple of (true_samples, false_samples)
        """
        import random
        from tqdm import tqdm
        
        random.seed(SEED)
        
        # Collect unique terms to translate
        terms_to_translate = set()
        for rel in relationships:
            terms_to_translate.add(rel['child_en'])
            terms_to_translate.add(rel['parent_en'])
        
        # Add some random terms for fake parents
        fake_parent_pool = random.sample(all_node_labels, min(len(all_node_labels), 5000))
        terms_to_translate.update(fake_parent_pool)
        
        terms_list = list(terms_to_translate)
        print(f"📝 Translating {len(terms_list)} unique terms...")
        
        # Translate in batches using internal method
        translation_map = {}
        for i in tqdm(range(0, len(terms_list), BATCH_SIZE), desc="Translating"):
            batch = terms_list[i:i + BATCH_SIZE]
            translations = self._translate_batch_internal(batch)
            for term, trans in zip(batch, translations):
                translation_map[term] = trans
        
        print(f"✅ Translated {len(translation_map)} terms")
        
        # Generate QA pairs
        true_samples = []
        false_samples = []
        
        for rel in tqdm(relationships, desc="Generating QA pairs"):
            child_en = rel['child_en']
            parent_en = rel['parent_en']
            child_vi = translation_map.get(child_en, child_en)
            parent_vi = translation_map.get(parent_en, parent_en)
            
            # 1. Generate TRUE sample with random instruction
            template = random.choice(TEMPLATES_TRUE)
            instruction = random.choice(INSTRUCTIONS)
            question = template.format(
                child_vi=child_vi, child_en=child_en,
                parent_vi=parent_vi, parent_en=parent_en
            )
            
            true_samples.append({
                "instruction": instruction,
                "input": question,
                "output": "Đúng",
                "question_type": "true_relationship",
                "child_en": child_en,
                "child_vi": child_vi,
                "parent_en": parent_en,
                "parent_vi": parent_vi
            })
            
            # 2. Generate FALSE sample with random fake parent and instruction
            fake_parent_en = random.choice(fake_parent_pool)
            while fake_parent_en == parent_en or fake_parent_en == child_en:
                fake_parent_en = random.choice(fake_parent_pool)
            
            fake_parent_vi = translation_map.get(fake_parent_en, fake_parent_en)
            
            template_false = random.choice(TEMPLATES_FALSE)
            instruction_false = random.choice(INSTRUCTIONS)
            question_false = template_false.format(
                child_vi=child_vi, child_en=child_en,
                fake_parent_vi=fake_parent_vi, fake_parent_en=fake_parent_en
            )
            
            false_samples.append({
                "instruction": instruction_false,
                "input": question_false,
                "output": "Sai",
                "question_type": "false_relationship",
                "child_en": child_en,
                "child_vi": child_vi,
                "fake_parent_en": fake_parent_en,
                "fake_parent_vi": fake_parent_vi
            })
        
        return true_samples, false_samples


@app.function(
    image=modal.Image.debian_slim().pip_install("tqdm"),
    volumes={"/data": volume},
    timeout=600
)
def load_hpo_data() -> Tuple[List[Dict], List[str]]:
    """Load HPO data and extract relationships"""
    import json
    import random
    
    random.seed(SEED)
    
    print("📂 Loading HPO data from volume...")
    
    # Try to load from volume first, then from local
    hpo_path = "/data/hp.json"
    
    with open(hpo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    graphs = data['graphs'][0]
    nodes = {n['id']: n for n in graphs['nodes']}
    edges = graphs['edges']
    
    # Get all node labels for fake parent selection
    all_labels = [n.get('lbl', '') for n in graphs['nodes'] if n.get('lbl')]
    
    # Filter valid is_a relationships
    relationships = []
    for edge in edges:
        if edge.get('pred') != 'is_a':
            continue
            
        child_id = edge['sub']
        parent_id = edge['obj']
        
        if child_id not in nodes or parent_id not in nodes:
            continue
            
        child_en = nodes[child_id].get('lbl', '')
        parent_en = nodes[parent_id].get('lbl', '')
        
        if not child_en or not parent_en:
            continue
        
        # Skip root term
        if "All" in parent_en or parent_id.endswith("HP_0000001"):
            continue
            
        relationships.append({
            'child_id': child_id,
            'child_en': child_en,
            'parent_id': parent_id,
            'parent_en': parent_en
        })
    
    print(f"📊 Found {len(relationships)} valid relationships")
    
    # Sample relationships based on mode
    random.shuffle(relationships)
    
    if TEST_MODE:
        sampled = relationships[:TEST_BATCH_SIZE]
        print(f"🧪 TEST MODE: Using only {len(sampled)} relationships")
    else:
        sampled = relationships[:TARGET_PAIRS]
        print(f"📋 Sampled {len(sampled)} relationships for translation")
    
    return sampled, all_labels


@app.function(
    image=modal.Image.debian_slim().pip_install("tqdm"),
    volumes={"/data": volume},
    timeout=300
)
def save_results(true_samples: List[Dict], false_samples: List[Dict]):
    """Save translated samples to volume"""
    import json
    import random
    
    random.seed(SEED)
    
    # Combine and shuffle
    all_samples = true_samples + false_samples
    random.shuffle(all_samples)
    
    # Split train/test (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    
    # Save to volume
    train_path = f"/data/{OUTPUT_TRAIN}"
    test_path = f"/data/{OUTPUT_TEST}"
    stats_path = f"/data/{OUTPUT_STATS}"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    # Statistics
    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "true_samples": len(true_samples),
        "false_samples": len(false_samples),
        "balance_ratio": f"{len(true_samples)}:{len(false_samples)}",
        "templates_used": {
            "true": len(TEMPLATES_TRUE),
            "false": len(TEMPLATES_FALSE)
        },
        "translation_model": "facebook/nllb-200-distilled-600M",
        "bilingual_format": "Vietnamese (English)"
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(train_samples)} train samples to {train_path}")
    print(f"✅ Saved {len(test_samples)} test samples to {test_path}")
    print(f"✅ Saved statistics to {stats_path}")
    
    # Commit volume
    volume.commit()
    
    return stats


@app.local_entrypoint()
def main():
    """Main entry point for Modal execution"""
    mode_str = "🧪 TEST MODE" if TEST_MODE else "🚀 PRODUCTION MODE"
    target = TEST_BATCH_SIZE if TEST_MODE else TARGET_PAIRS
    
    print(f"{mode_str} - HPO Bilingual Translation Pipeline")
    print(f"📊 Target: {target} pairs = {target * 2} total samples")
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    # Step 1: Load HPO data
    print("\n📂 Step 1: Loading HPO data...")
    relationships, all_labels = load_hpo_data.remote()
    print(f"   Loaded {len(relationships)} relationships")
    
    # Step 2: Translate and generate QA pairs
    print("\n🔄 Step 2: Translating and generating QA pairs...")
    translator = BilingualTranslator()
    true_samples, false_samples = translator.translate_relationships.remote(
        relationships, 
        all_labels
    )
    print(f"   Generated {len(true_samples)} true + {len(false_samples)} false samples")
    
    # Step 3: Save results
    print("\n💾 Step 3: Saving results...")
    stats = save_results.remote(true_samples, false_samples)
    
    print("\n" + "="*50)
    print("🎉 HPO Bilingual Translation Complete!")
    print("="*50)
    print(f"Mode: {mode_str}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Train: {stats['train_samples']}, Test: {stats['test_samples']}")
    print(f"Balance: {stats['balance_ratio']}")
    print("\n📥 Download with:")
    print(f"   modal volume get medical-data {OUTPUT_TRAIN} ./")
    print(f"   modal volume get medical-data {OUTPUT_TEST} ./")
    print(f"   modal volume get medical-data {OUTPUT_STATS} ./")


# For testing locally without Modal
def test_templates():
    """Test template formatting"""
    sample_data = {
        "child_vi": "Chứng rậm lông",
        "child_en": "Hirsutism",
        "parent_vi": "Rậm lông toàn thân",
        "parent_en": "Generalized hirsutism",
        "fake_parent_vi": "Đau đầu",
        "fake_parent_en": "Headache"
    }
    
    print("=== TRUE Templates ===")
    for tmpl in TEMPLATES_TRUE:
        print(tmpl.format(**sample_data))
    
    print("\n=== FALSE Templates ===")
    for tmpl in TEMPLATES_FALSE:
        print(tmpl.format(**sample_data))


if __name__ == "__main__":
    # Local test
    test_templates()
