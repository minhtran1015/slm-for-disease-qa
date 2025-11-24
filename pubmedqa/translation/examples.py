"""
Example Usage Script
====================
Demonstrates various ways to use the translation pipeline and utilities.
"""

import sys
from pathlib import Path

# Add translation directory to path
sys.path.append(str(Path(__file__).parent))

from translate_pipeline import UMLSMedicalTranslator
from umls_mapper import UMLSMapper
from translator import VinAITranslator


def example_1_full_pipeline():
    """Example 1: Full pipeline with default settings."""
    print("\n" + "="*60)
    print("Example 1: Full Pipeline Translation")
    print("="*60 + "\n")
    
    translator = UMLSMedicalTranslator()
    
    # Process 10 samples for testing
    translator.process_pubmedqa_dataset(
        input_file="../data/pqaa_train_set.json",
        output_file="./outputs/example1_output.json",
        max_samples=10
    )


def example_2_umls_only():
    """Example 2: Extract UMLS concepts without translation."""
    print("\n" + "="*60)
    print("Example 2: UMLS Concept Extraction Only")
    print("="*60 + "\n")
    
    mapper = UMLSMapper(linker_name="umls")
    
    sample_text = """
    Patient presents with acute myocardial infarction and atrial fibrillation.
    Medical history includes hypertension and type 2 diabetes mellitus.
    Currently on statins and metformin therapy.
    """
    
    concepts = mapper.extract_concepts(sample_text)
    
    print(f"Input:\n{sample_text}\n")
    print(f"Extracted {len(concepts)} medical concepts:\n")
    print(mapper.format_detailed_glossary(concepts))


def example_3_translation_only():
    """Example 3: Translation without UMLS mapping."""
    print("\n" + "="*60)
    print("Example 3: Translation Only (No UMLS)")
    print("="*60 + "\n")
    
    translator = VinAITranslator()
    
    texts = [
        "Does aspirin reduce the risk of heart attack?",
        "What are the symptoms of diabetes?",
        "Is surgery necessary for appendicitis?"
    ]
    
    print("Translating medical questions...\n")
    
    translations = translator.translate_batch(texts)
    
    for i, (original, translation) in enumerate(zip(texts, translations), 1):
        print(f"{i}. EN: {original}")
        print(f"   VI: {translation}\n")


def example_4_custom_config():
    """Example 4: Custom configuration."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60 + "\n")
    
    # Use MeSH instead of UMLS, lower confidence threshold
    translator = UMLSMedicalTranslator(config_path=None)
    translator.config.update({
        "linker_name": "mesh",
        "confidence_threshold": 0.5,
        "batch_size": 8
    })
    
    # Reload with new config
    translator.nlp = translator._load_scispacy()
    
    print(f"Using configuration:")
    print(f"  Linker: {translator.config['linker_name']}")
    print(f"  Confidence: {translator.config['confidence_threshold']}")
    print(f"  Batch Size: {translator.config['batch_size']}\n")


def example_5_alternative_translations():
    """Example 5: Generate multiple translation alternatives."""
    print("\n" + "="*60)
    print("Example 5: Alternative Translations")
    print("="*60 + "\n")
    
    translator = VinAITranslator()
    
    text = "Do statins prevent cardiovascular disease?"
    
    alternatives = translator.translate_with_alternatives(text, num_alternatives=3)
    
    print(f"Original: {text}\n")
    print("Alternative translations:")
    for alt in alternatives:
        print(f"  {alt['rank']}. {alt['translation']}")


def example_6_batch_processing():
    """Example 6: Efficient batch processing."""
    print("\n" + "="*60)
    print("Example 6: Large Batch Processing")
    print("="*60 + "\n")
    
    translator = VinAITranslator(batch_size=32)
    
    # Simulate large dataset
    questions = [
        f"What is the treatment for disease {i}?"
        for i in range(100)
    ]
    
    print(f"Translating {len(questions)} questions in batches of 32...\n")
    
    translations = translator.translate_large_batch(questions)
    
    print(f"✅ Translated {len(translations)} questions")
    print(f"First 3 results:")
    for i in range(3):
        print(f"  {i+1}. {translations[i]}")


def example_7_umls_statistics():
    """Example 7: Analyze UMLS concept statistics."""
    print("\n" + "="*60)
    print("Example 7: UMLS Concept Statistics")
    print("="*60 + "\n")
    
    mapper = UMLSMapper()
    
    medical_texts = [
        "Patient has coronary artery disease and hypertension.",
        "Diagnosed with type 2 diabetes mellitus and hyperlipidemia.",
        "Presenting with acute respiratory distress syndrome.",
        "Treatment includes insulin therapy and metformin.",
        "Chest X-ray shows pulmonary edema and cardiomegaly."
    ]
    
    all_concepts = []
    for text in medical_texts:
        concepts = mapper.extract_concepts(text)
        all_concepts.extend(concepts)
    
    stats = mapper.get_statistics(all_concepts)
    
    print(f"Analyzed {len(medical_texts)} medical texts\n")
    print("Statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"\nTop entity types:")
    for etype, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype}: {count}")


def main():
    """Run all examples."""
    examples = [
        ("Full Pipeline", example_1_full_pipeline),
        ("UMLS Only", example_2_umls_only),
        ("Translation Only", example_3_translation_only),
        ("Custom Config", example_4_custom_config),
        ("Alternatives", example_5_alternative_translations),
        ("Batch Processing", example_6_batch_processing),
        ("Statistics", example_7_umls_statistics)
    ]
    
    print("\n" + "="*60)
    print("🎓 Translation Pipeline Examples")
    print("="*60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun specific example: python examples.py <number>")
    print("Run all examples: python examples.py all\n")
    
    if len(sys.argv) < 2:
        print("Please specify example number or 'all'")
        return
    
    arg = sys.argv[1]
    
    if arg == "all":
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"❌ Example '{name}' failed: {e}")
    else:
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
        except ValueError:
            print("Invalid argument. Use a number or 'all'")


if __name__ == "__main__":
    main()
