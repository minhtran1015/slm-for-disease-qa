"""
UMLS-Aware Medical Translation Pipeline for PubMedQA
=====================================================
This pipeline combines Knowledge Graph Extraction (UMLS/MeSH) with Neural Machine Translation (VinAI)
to translate medical Q&A data while maintaining medical terminology accuracy.

Pipeline Steps:
1. Entity Extraction: Scan English text for medical terms using Scispacy
2. Concept Mapping (UMLS): Resolve terms to CUI (Concept Unique Identifier) with canonical names
3. Term Translation: Translate canonical names for medical accuracy
4. Sentence Translation: Full sentence translation using VinAI
5. Output: Translation + Medical Glossary for validation

Author: Research Project - SLM for Disease QA
Date: 2025-11-24
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import spacy
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class UMLSMedicalTranslator:
    """
    Main translator class that combines UMLS concept mapping with VinAI translation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the translator with configuration.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing UMLS-Aware Medical Translator on {self.device}...")
        
        # Load models
        self.nlp = self._load_scispacy()
        self.tokenizer, self.model = self._load_vinai()
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "total_entities": 0,
            "total_mapped": 0,
            "translation_time": 0,
            "umls_time": 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "model_name": "vinai/vinai-translate-en2vi",
            "scispacy_model": "en_core_sci_sm",
            "linker_name": "umls",  # Can be 'mesh' for MeSH-only
            "batch_size": 16,
            "max_length": 512,
            "num_beams": 4,
            "resolve_abbreviations": True,
            "confidence_threshold": 0.7,
            "cache_dir": "./models_cache"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _load_scispacy(self):
        """Load Scispacy model with UMLS entity linker."""
        print(f"📚 Loading Scispacy ({self.config['scispacy_model']})...")
        
        try:
            nlp = spacy.load(self.config['scispacy_model'])
        except OSError:
            print("❌ Scispacy model not found. Please run setup_translation.sh first!")
            raise
        
        # Add UMLS/MeSH linker
        print(f"🔗 Adding {self.config['linker_name'].upper()} entity linker...")
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": self.config["resolve_abbreviations"],
                "linker_name": self.config["linker_name"]
            }
        )
        
        return nlp
    
    def _load_vinai(self):
        """Load VinAI translation model."""
        print(f"🇻🇳 Loading VinAI model ({self.config['model_name']})...")
        
        cache_dir = self.config.get("cache_dir", "./models_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            src_lang="en_XX",
            cache_dir=cache_dir
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["model_name"],
            cache_dir=cache_dir
        ).to(self.device)
        
        model.eval()  # Set to evaluation mode
        
        return tokenizer, model
    
    def extract_umls_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities and map to UMLS concepts.
        
        Args:
            text: Input English text
            
        Returns:
            List of concept dictionaries with CUI, canonical name, and scores
        """
        doc = self.nlp(text)
        concepts = []
        
        linker = self.nlp.get_pipe("scispacy_linker")
        
        for entity in doc.ents:
            if entity._.kb_ents:
                # Get top matching concept
                cui, score = entity._.kb_ents[0]
                
                # Only include high-confidence mappings
                if score < self.config.get("confidence_threshold", 0.7):
                    continue
                
                # Get canonical name from knowledge base
                concept_entity = linker.kb.cui_to_entity[cui]
                canonical_name = concept_entity.canonical_name
                
                # Get all aliases for reference
                aliases = list(concept_entity.aliases)[:3]  # Top 3 aliases
                
                concepts.append({
                    "original_text": entity.text,
                    "start": entity.start_char,
                    "end": entity.end_char,
                    "cui": cui,
                    "canonical_name": canonical_name,
                    "confidence": float(score),
                    "aliases": aliases,
                    "entity_type": entity.label_
                })
                
                self.stats["total_entities"] += 1
                self.stats["total_mapped"] += 1
        
        return concepts
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts using VinAI model.
        
        Args:
            texts: List of English texts
            
        Returns:
            List of Vietnamese translations
        """
        if not texts:
            return []
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                decoder_start_token_id=self.tokenizer.lang_code_to_id["vi_VN"],
                num_return_sequences=1,
                num_beams=self.config["num_beams"],
                early_stopping=True
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return translations
    
    def create_medical_glossary(self, concepts: List[Dict[str, Any]]) -> str:
        """
        Create a formatted medical glossary string for teacher verification.
        
        Args:
            concepts: List of UMLS concepts
            
        Returns:
            Formatted glossary string
        """
        if not concepts:
            return "No medical terms identified"
        
        glossary_items = []
        for concept in concepts:
            item = f"{concept['original_text']} → {concept['canonical_name']} (CUI:{concept['cui']}, conf:{concept['confidence']:.2f})"
            glossary_items.append(item)
        
        return " | ".join(glossary_items)
    
    def process_pubmedqa_dataset(
        self,
        input_file: str,
        output_file: str,
        max_samples: Optional[int] = None
    ):
        """
        Process PubMedQA dataset with UMLS mapping and translation.
        
        Args:
            input_file: Path to input JSON file (PubMedQA format)
            output_file: Path to output JSON file
            max_samples: Maximum number of samples to process (None for all)
        """
        print(f"\n{'='*60}")
        print(f"📖 Processing PubMedQA Dataset")
        print(f"{'='*60}")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        if max_samples:
            print(f"Limit:  {max_samples} samples")
        print(f"{'='*60}\n")
        
        # Load dataset
        print("📂 Loading dataset...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different PubMedQA formats
        if isinstance(data, dict):
            # Format: {PMID: {QUESTION, CONTEXTS, ...}}
            pmids = list(data.keys())
            if max_samples:
                pmids = pmids[:max_samples]
            
            samples = [(pmid, data[pmid]) for pmid in pmids]
        elif isinstance(data, list):
            # Format: [{question, context, ...}]
            if max_samples:
                data = data[:max_samples]
            samples = [(f"sample_{i}", item) for i, item in enumerate(data)]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        self.stats["total_samples"] = len(samples)
        print(f"✅ Loaded {len(samples)} samples\n")
        
        # Process in pipeline stages
        results = {}
        
        # Stage 1: UMLS Concept Extraction
        print("🔍 Stage 1/3: Extracting UMLS Concepts...")
        umls_mappings = {}
        
        for pmid, sample in tqdm(samples, desc="UMLS Extraction"):
            question = sample.get("QUESTION", sample.get("question", ""))
            contexts = sample.get("CONTEXTS", sample.get("contexts", []))
            
            # Combine question and contexts for entity extraction
            full_text = question
            if contexts:
                context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
                full_text = f"{question} {context_text}"
            
            # Extract concepts
            concepts = self.extract_umls_concepts(full_text)
            
            umls_mappings[pmid] = {
                "question": question,
                "contexts": contexts,
                "concepts": concepts,
                "glossary": self.create_medical_glossary(concepts),
                "final_decision": sample.get("final_decision", sample.get("answer", ""))
            }
        
        print(f"✅ Extracted {self.stats['total_entities']} entities, mapped {self.stats['total_mapped']} to UMLS\n")
        
        # Stage 2: Translate Questions
        print("🌍 Stage 2/3: Translating Questions...")
        questions = [umls_mappings[pmid]["question"] for pmid in [s[0] for s in samples]]
        translated_questions = []
        
        batch_size = self.config["batch_size"]
        for i in tqdm(range(0, len(questions), batch_size), desc="Question Translation"):
            batch = questions[i:i+batch_size]
            translated_questions.extend(self.translate_batch(batch))
        
        # Update results
        for idx, pmid in enumerate([s[0] for s in samples]):
            umls_mappings[pmid]["question_vi"] = translated_questions[idx]
        
        print(f"✅ Translated {len(translated_questions)} questions\n")
        
        # Stage 3: Translate Contexts
        print("📄 Stage 3/3: Translating Contexts...")
        
        for pmid in tqdm([s[0] for s in samples], desc="Context Translation"):
            contexts = umls_mappings[pmid]["contexts"]
            
            if not contexts:
                umls_mappings[pmid]["contexts_vi"] = []
                continue
            
            if isinstance(contexts, list):
                # Translate each context sentence
                contexts_vi = []
                for i in range(0, len(contexts), batch_size):
                    batch = contexts[i:i+batch_size]
                    contexts_vi.extend(self.translate_batch(batch))
                umls_mappings[pmid]["contexts_vi"] = contexts_vi
            else:
                # Single context string
                context_vi = self.translate_batch([str(contexts)])[0]
                umls_mappings[pmid]["contexts_vi"] = context_vi
        
        print(f"✅ Translated contexts for {len(samples)} samples\n")
        
        # Format output
        print("💾 Formatting output...")
        output_data = {}
        label_map = {"yes": "có", "no": "không", "maybe": "có thể"}
        
        for pmid in [s[0] for s in samples]:
            mapping = umls_mappings[pmid]
            
            output_data[pmid] = {
                "QUESTION": mapping["question"],
                "QUESTION_VI": mapping["question_vi"],
                "CONTEXTS": mapping["contexts"],
                "CONTEXTS_VI": mapping["contexts_vi"],
                "ANSWER": mapping["final_decision"],
                "ANSWER_VI": label_map.get(mapping["final_decision"], mapping["final_decision"]),
                "UMLS_CONCEPTS": mapping["concepts"],
                "MEDICAL_GLOSSARY": mapping["glossary"],
                "TRANSLATION_METADATA": {
                    "model": self.config["model_name"],
                    "linker": self.config["linker_name"],
                    "num_concepts": len(mapping["concepts"]),
                    "translated_at": datetime.now().isoformat()
                }
            }
        
        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved to {output_file}\n")
        
        # Print statistics
        self._print_statistics(output_data)
        
        # Print example
        self._print_example(output_data)
    
    def _print_statistics(self, output_data: Dict[str, Any]):
        """Print processing statistics."""
        print(f"\n{'='*60}")
        print(f"📊 Processing Statistics")
        print(f"{'='*60}")
        print(f"Total Samples:       {self.stats['total_samples']}")
        print(f"Total Entities:      {self.stats['total_entities']}")
        print(f"Mapped to UMLS:      {self.stats['total_mapped']}")
        
        if self.stats['total_entities'] > 0:
            mapping_rate = (self.stats['total_mapped'] / self.stats['total_entities']) * 100
            print(f"Mapping Success:     {mapping_rate:.1f}%")
        
        avg_concepts = sum(len(v["UMLS_CONCEPTS"]) for v in output_data.values()) / len(output_data)
        print(f"Avg Concepts/Sample: {avg_concepts:.2f}")
        print(f"{'='*60}\n")
    
    def _print_example(self, output_data: Dict[str, Any]):
        """Print example translation for verification."""
        print(f"\n{'='*60}")
        print(f"📝 Example Translation (for Teacher Verification)")
        print(f"{'='*60}")
        
        # Get first sample
        pmid = list(output_data.keys())[0]
        sample = output_data[pmid]
        
        print(f"PMID: {pmid}\n")
        print(f"Original Question (EN):")
        print(f"  {sample['QUESTION']}\n")
        print(f"UMLS Medical Glossary:")
        print(f"  {sample['MEDICAL_GLOSSARY']}\n")
        print(f"Translated Question (VI):")
        print(f"  {sample['QUESTION_VI']}\n")
        print(f"Answer: {sample['ANSWER']} → {sample['ANSWER_VI']}\n")
        print(f"Medical Concepts Identified: {sample['TRANSLATION_METADATA']['num_concepts']}")
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UMLS-Aware Medical Translation Pipeline for PubMedQA"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/pqaa_train_set.json",
        help="Path to input PubMedQA JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/pubmedqa_vi_umls.json",
        help="Path to output translated JSON file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./translation_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = UMLSMedicalTranslator(config_path=args.config)
    
    # Process dataset
    translator.process_pubmedqa_dataset(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples
    )
    
    print("✅ Pipeline completed successfully!")
    print(f"📁 Output saved to: {args.output}")


if __name__ == "__main__":
    main()
