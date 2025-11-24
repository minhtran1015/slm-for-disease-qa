"""
UMLS Mapper Utility Module
===========================
Standalone module for extracting medical entities and mapping them to UMLS/MeSH concepts.
Can be used independently of the main translation pipeline.

Usage:
    from umls_mapper import UMLSMapper
    
    mapper = UMLSMapper(linker_name="umls")
    concepts = mapper.extract_concepts("Patient has atrial fibrillation and hypertension")
    
    # Print formatted glossary
    print(mapper.format_glossary(concepts))
"""

import json
from typing import List, Dict, Any, Optional
import spacy
from scispacy.linking import EntityLinker


class UMLSMapper:
    """
    Medical entity extractor with UMLS/MeSH concept mapping capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        linker_name: str = "umls",
        confidence_threshold: float = 0.7,
        resolve_abbreviations: bool = True
    ):
        """
        Initialize the UMLS mapper.
        
        Args:
            model_name: Scispacy model name (en_core_sci_sm/md/lg)
            linker_name: Entity linker to use (umls, mesh, rxnorm)
            confidence_threshold: Minimum confidence score for mappings (0.0-1.0)
            resolve_abbreviations: Whether to resolve medical abbreviations
        """
        self.model_name = model_name
        self.linker_name = linker_name
        self.confidence_threshold = confidence_threshold
        self.resolve_abbreviations = resolve_abbreviations
        
        # Load Scispacy
        print(f"Loading Scispacy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        
        # Add entity linker
        print(f"Adding {linker_name.upper()} entity linker...")
        self.nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": resolve_abbreviations,
                "linker_name": linker_name
            }
        )
        
        self.linker = self.nlp.get_pipe("scispacy_linker")
        
        print(f"✅ UMLS Mapper initialized with {linker_name.upper()}")
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities and map to UMLS/MeSH concepts.
        
        Args:
            text: Input medical text
            
        Returns:
            List of concept dictionaries with CUI, canonical names, and metadata
        """
        doc = self.nlp(text)
        concepts = []
        
        for entity in doc.ents:
            if not entity._.kb_ents:
                continue
            
            # Get top matching concept
            cui, score = entity._.kb_ents[0]
            
            # Filter by confidence
            if score < self.confidence_threshold:
                continue
            
            # Get concept details from knowledge base
            concept_entity = self.linker.kb.cui_to_entity[cui]
            
            concept_info = {
                "original_text": entity.text,
                "start": entity.start_char,
                "end": entity.end_char,
                "cui": cui,
                "canonical_name": concept_entity.canonical_name,
                "confidence": float(score),
                "entity_type": entity.label_,
                "aliases": list(concept_entity.aliases)[:5],  # Top 5 aliases
                "definition": getattr(concept_entity, 'definition', ''),
                "semantic_types": list(getattr(concept_entity, 'types', []))
            }
            
            concepts.append(concept_info)
        
        return concepts
    
    def format_glossary(self, concepts: List[Dict[str, Any]]) -> str:
        """
        Format concepts into a readable glossary string.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Formatted glossary string
        """
        if not concepts:
            return "No medical terms identified"
        
        glossary_items = []
        for concept in concepts:
            item = (
                f"{concept['original_text']} → {concept['canonical_name']} "
                f"(CUI:{concept['cui']}, conf:{concept['confidence']:.2f})"
            )
            glossary_items.append(item)
        
        return " | ".join(glossary_items)
    
    def format_detailed_glossary(self, concepts: List[Dict[str, Any]]) -> str:
        """
        Format concepts into a detailed glossary with definitions.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Detailed formatted glossary string
        """
        if not concepts:
            return "No medical terms identified"
        
        lines = []
        for i, concept in enumerate(concepts, 1):
            lines.append(f"\n{i}. {concept['original_text']}")
            lines.append(f"   → Canonical Name: {concept['canonical_name']}")
            lines.append(f"   → CUI: {concept['cui']}")
            lines.append(f"   → Confidence: {concept['confidence']:.2f}")
            lines.append(f"   → Entity Type: {concept['entity_type']}")
            
            if concept['aliases']:
                lines.append(f"   → Aliases: {', '.join(concept['aliases'][:3])}")
            
            if concept.get('definition'):
                lines.append(f"   → Definition: {concept['definition'][:200]}...")
        
        return "\n".join(lines)
    
    def export_concepts_json(self, concepts: List[Dict[str, Any]], output_file: str):
        """
        Export concepts to JSON file.
        
        Args:
            concepts: List of concept dictionaries
            output_file: Path to output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concepts, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Exported {len(concepts)} concepts to {output_file}")
    
    def batch_extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract concepts from multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of concept lists (one per input text)
        """
        all_concepts = []
        
        for text in texts:
            concepts = self.extract_concepts(text)
            all_concepts.append(concepts)
        
        return all_concepts
    
    def get_statistics(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for extracted concepts.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not concepts:
            return {
                "total_concepts": 0,
                "avg_confidence": 0.0,
                "entity_types": {},
                "semantic_types": {}
            }
        
        # Count entity types
        entity_type_counts = {}
        for concept in concepts:
            etype = concept['entity_type']
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1
        
        # Count semantic types
        semantic_type_counts = {}
        for concept in concepts:
            for stype in concept.get('semantic_types', []):
                semantic_type_counts[stype] = semantic_type_counts.get(stype, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(c['confidence'] for c in concepts) / len(concepts)
        
        return {
            "total_concepts": len(concepts),
            "avg_confidence": avg_confidence,
            "entity_types": entity_type_counts,
            "semantic_types": dict(list(semantic_type_counts.items())[:10])  # Top 10
        }


def main():
    """Demo usage of UMLSMapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UMLS Medical Entity Mapper")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File containing text (one line)")
    parser.add_argument("--linker", type=str, default="umls", choices=["umls", "mesh", "rxnorm"])
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--output", type=str, help="Output JSON file for concepts")
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = "Patient presents with atrial fibrillation and hypertension. Prescribed statins for cholesterol management."
        print(f"Using demo text: {text}\n")
    
    # Initialize mapper
    mapper = UMLSMapper(
        linker_name=args.linker,
        confidence_threshold=args.confidence
    )
    
    # Extract concepts
    print(f"\n{'='*60}")
    print("Extracting Medical Concepts")
    print(f"{'='*60}\n")
    
    concepts = mapper.extract_concepts(text)
    
    # Print results
    print(f"Input Text:\n{text}\n")
    print(f"\nExtracted Concepts: {len(concepts)}\n")
    print(mapper.format_detailed_glossary(concepts))
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}")
    stats = mapper.get_statistics(concepts)
    print(json.dumps(stats, indent=2))
    
    # Export if requested
    if args.output:
        mapper.export_concepts_json(concepts, args.output)


if __name__ == "__main__":
    main()
