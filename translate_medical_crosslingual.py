"""
Cross-Lingual Transfer Learning (CLTL) Medical Data Translation
Keeps English context, translates questions/labels to Vietnamese for international knowledge transfer
"""

import modal
import json
import random
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import re

app = modal.App("medical-cltl-translation")

# Local translation model image with GPU support and optimizations
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "transformers>=4.35.0",
        "torch>=2.0.0", 
        "sentencepiece",
        "accelerate",
        "aiofiles",
        "datasets",
        "optimum[onnxruntime-gpu]"
    ])
)

volume = modal.Volume.from_name("medical-data", create_if_missing=True)

# Vietnamese instruction templates for CLTL
CLTL_INSTRUCTIONS = [
    "Dựa vào bối cảnh y khoa quốc tế, hãy trả lời câu hỏi bằng tiếng Việt. Trả lời Đúng hoặc Sai.",
    "Sử dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt. Chỉ trả lời Đúng hoặc Sai.",
    "Phân tích thông tin y khoa từ tài liệu quốc tế và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai.",
    "Dựa trên bằng chứng y khoa quốc tế, đánh giá câu hỏi tiếng Việt và trả lời Đúng hoặc Sai.",
    "Áp dụng kiến thức y học chuẩn quốc tế để trả lời câu hỏi tiếng Việt. Trả lời Đúng hoặc Sai.",
    "Chuyển giao tri thức y khoa quốc tế sang tư duy tiếng Việt. Trả lời Đúng hoặc Sai.",
    "Sử dụng cơ sở dữ liệu y khoa quốc tế để phân tích và trả lời câu hỏi tiếng Việt bằng Đúng hoặc Sai.",
    "Kết hợp kiến thức y học quốc tế với ngôn ngữ Việt Nam để trả lời. Chọn Đúng hoặc Sai."
]

@app.function(image=image, gpu="A10G", timeout=1800, retries=3)
async def translate_cltl_batch(
    samples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Translate medical samples for Cross-Lingual Transfer Learning
    Optimized for accuracy and GPU efficiency
    
    Uses NLLB-200 (No Language Left Behind) for state-of-the-art medical translation accuracy
    Superior to M2M100: better accuracy, 200+ languages, optimized for low-resource pairs
    Includes mixed precision inference and beam search
    
    Args:
        samples: List of medical samples with question, context, answer
    
    Returns:
        List of CLTL samples with Vietnamese questions/labels
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    
    # GPU optimization settings
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    # Use NLLB-200 as primary (best quality for Vietnamese medical translation)
    # Fallback chain: NLLB-200 -> Helsinki-NLP (skip VinAI due to repetition issues)
    model_loaded = False
    model_name = None
    model_type = None
    
    # Try NLLB-200 first (best quality for Vietnamese)
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        tokenizer.src_lang = "eng_Latn"
        model_type = "nllb"
        model_loaded = True
        print(f"✅ Loaded NLLB-200 distilled model")
    except Exception as e:
        print(f"NLLB-200 load failed: {e}, trying Helsinki-NLP...")
    
    # Final fallback to Helsinki-NLP
    if not model_loaded:
        model_name = "Helsinki-NLP/opus-mt-en-vi"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        model_type = "helsinki"
        print(f"✅ Loaded Helsinki-NLP model")
    
    results = []
    
    with torch.no_grad():
        for sample in samples:
            try:
                question = sample.get("question", "")
                context = sample.get("context", "")
                answer = sample.get("answer", "")
                
                if not question or not answer:
                    continue
                
                # Translate question to Vietnamese
                vietnamese_question = translate_question_m2m100(tokenizer, model, question, model_type)
                
                # CLTL: Always keep English context for cross-lingual knowledge transfer
                english_context = validate_english_context(context) if context else context
                
                # Convert answer to Vietnamese
                vietnamese_answer = "Đúng" if answer.lower() in ['yes', 'true', '1'] else "Sai"
                
                # Create CLTL format
                cltl_sample = create_cltl_format(
                    instruction=random.choice(CLTL_INSTRUCTIONS),
                    vietnamese_question=vietnamese_question,
                    english_context=english_context,
                    vietnamese_context=None,
                    vietnamese_answer=vietnamese_answer,
                    original_data=sample
                )
                
                results.append(cltl_sample)
                
            except Exception as e:
                print(f"CLTL translation error: {e}")
                continue
    
    return results

async def translate_question_to_vietnamese(translator, question: str) -> str:
    """Translate English medical question to natural Vietnamese using local model"""
    
    try:
        # Use local translation model
        result = translator(question, max_length=200)
        vietnamese_question = result[0]['translation_text']
        
        # Post-process to ensure medical question format
        if not vietnamese_question.endswith('?'):
            vietnamese_question += '?'
            
        return vietnamese_question
    except Exception as e:
        print(f"Translation error for question: {question} - {e}")
        # Fallback to simple template-based translation
        return f"Có phải {question.lower().replace('?', '')} không?"

def translate_question_m2m100(tokenizer, model, question: str, model_type: str = "nllb") -> str:
    """Translate English medical question to Vietnamese using the loaded model"""
    
    try:
        import torch
        
        # Tokenize based on model type
        inputs = tokenizer(question, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if model_type == "vinai":
                # VinAI needs repetition penalty to avoid repetitive output
                translated_ids = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
            elif model_type == "nllb":
                # NLLB uses language token IDs
                vie_token_id = tokenizer.convert_tokens_to_ids("vie_Latn")
                translated_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=vie_token_id,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
            else:  # helsinki
                translated_ids = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
        
        vietnamese_question = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        if not vietnamese_question.endswith('?'):
            vietnamese_question += '?'
            
        return vietnamese_question
    except Exception as e:
        print(f"Translation error ({model_type}): {e}")
        return f"Có phải {question.lower().replace('?', '')} không?"

def validate_english_context(context: str) -> str:
    """Validate and clean English medical context for CLTL - non-async version"""
    
    try:
        # Limit context length for processing
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        # Keep original English context for CLTL
        # This preserves international medical knowledge accuracy
        return context.strip()
        
    except Exception as e:
        print(f"Context validation error: {e}")
        return context

def create_cltl_format(
    instruction: str,
    vietnamese_question: str,
    english_context: str,
    vietnamese_context: Optional[str],
    vietnamese_answer: str,
    original_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create Cross-Lingual Transfer Learning format
    
    Formats data for bilingual medical knowledge transfer
    """
    
    # Base CLTL format
    cltl_sample = {
        "instruction": instruction,
        "input": vietnamese_question,
        "output": vietnamese_answer,
        
        # Cross-lingual context preservation
        "context_english": english_context,
        "question_vietnamese": vietnamese_question,
        "answer_vietnamese": vietnamese_answer,
        
        # Metadata for CLTL research
        "cltl_type": "medical_qa",
        "source_language": "english",
        "target_language": "vietnamese",
        "knowledge_domain": "medical"
    }
    
    # Add Vietnamese context if translated
    if vietnamese_context:
        cltl_sample["context_vietnamese"] = vietnamese_context
        cltl_sample["context_bilingual"] = "true"
    else:
        cltl_sample["context_bilingual"] = "false"
    
    # Add original metadata
    if "pmid" in original_data:
        cltl_sample["pmid"] = original_data["pmid"]
    if "meshes" in original_data:
        cltl_sample["medical_concepts"] = original_data["meshes"]
    if "year" in original_data:
        cltl_sample["publication_year"] = original_data["year"]
    
    return cltl_sample

@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def extract_pubmedqa_cltl_samples(max_samples: int = 2000) -> List[Dict[str, Any]]:
    """
    Extract PubMedQA data for CLTL processing
    Preserves context and metadata for cross-lingual transfer
    Excludes all 'maybe' answers - keeps only yes/no
    """
    import json
    
    print("🔍 Extracting PubMedQA data for CLTL (yes/no only, excluding maybe)...")
    
    try:
        with open("/data/ori_pqal.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ PubMedQA file not found.")
        return []
    
    samples = []
    count = 0
    excluded_maybe = 0
    
    for pmid, item in data.items():
        if count >= max_samples:
            break
            
        question = item.get("QUESTION", "").strip()
        contexts = item.get("CONTEXTS", [])
        answer = item.get("final_decision", "").lower().strip()
        
        # FILTER: Exclude 'maybe' answers - only keep yes/no
        if answer == "maybe":
            excluded_maybe += 1
            continue
        
        if question and answer in ["yes", "no"] and contexts:
            # Combine first few context sentences
            context = " ".join(contexts[:3])  # First 3 sentences for context
            
            sample = {
                "question": question,
                "context": context,
                "answer": answer,
                "pmid": pmid,
                "meshes": item.get("MESHES", []),
                "year": item.get("YEAR", ""),
                "labels": item.get("LABELS", [])
            }
            
            samples.append(sample)
            count += 1
    
    print(f"✅ Extracted {len(samples)} PubMedQA yes/no samples for CLTL")
    print(f"⏭️  Excluded {excluded_maybe} maybe answers")
    return samples

@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def extract_bioasq_cltl_samples(max_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract BioASQ data for CLTL processing
    Filter to only YES/NO questions as per CLTL requirement
    """
    import json
    
    print("🔍 Extracting BioASQ data for CLTL (yesno questions only)...")
    
    try:
        with open("/data/training14b.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ BioASQ file not found.")
        return []
    
    samples = []
    count = 0
    skipped = 0
    
    questions_list = data.get("questions", [])
    
    for item in questions_list:
        if count >= max_samples:
            break
            
        question = item.get("body", "").strip()
        question_type = item.get("type", "")
        snippets = item.get("snippets", [])
        ideal_answers = item.get("ideal_answer", [])
        
        # FILTER: Only accept yesno questions
        if question_type != "yesno":
            skipped += 1
            continue
        
        if not question or not snippets:
            skipped += 1
            continue
        
        # Create context from snippets only
        context_parts = []
        for snippet in snippets[:2]:  # First 2 snippets
            text = snippet.get("text", "")
            if text:
                context_parts.append(text)
        
        context = " ".join(context_parts)
        
        if not context:
            skipped += 1
            continue
        
        # Determine yes/no answer
        answer = determine_yesno_from_bioasq(ideal_answers, snippets)
        
        if answer in ["yes", "no"]:
            sample = {
                "question": question,
                "context": context,
                "answer": answer,
                "bioasq_id": item.get("id", ""),
                "question_type": question_type,
                "concepts": item.get("concepts", [])
            }
            
            samples.append(sample)
            count += 1
    
    print(f"✅ Extracted {len(samples)} BioASQ yesno samples with context for CLTL")
    print(f"⏭️  Skipped {skipped} non-yesno questions")
    return samples

def determine_yesno_from_bioasq(ideal_answers: List[str], snippets: List[Dict]) -> str:
    """Enhanced yes/no determination for BioASQ"""
    
    text_sources = []
    text_sources.extend(ideal_answers)
    
    for snippet in snippets:
        if snippet.get("text"):
            text_sources.append(snippet["text"])
    
    if not text_sources:
        return random.choice(["yes", "no"])
    
    combined_text = " ".join(text_sources).lower()
    
    # Enhanced keyword matching
    positive_indicators = [
        "yes", "positive", "increased", "higher", "associated", "significant", 
        "effective", "beneficial", "successful", "confirmed", "demonstrated",
        "observed", "found", "showed", "indicated", "suggested", "evidence"
    ]
    
    negative_indicators = [
        "no", "not", "negative", "decreased", "lower", "unrelated", 
        "ineffective", "unsuccessful", "failed", "absence", "lack",
        "without", "unable", "cannot", "did not", "does not", "insufficient"
    ]
    
    positive_score = sum(1 for word in positive_indicators if word in combined_text)
    negative_score = sum(1 for word in negative_indicators if word in combined_text)
    
    if positive_score > negative_score:
        return "yes"
    elif negative_score > positive_score:
        return "no"
    else:
        return random.choice(["yes", "no"])

@app.function(image=image, volumes={"/data": volume}, gpu="A10G", timeout=7200)
def process_cltl_translation(
    max_pubmedqa: int = 2000,
    max_bioasq: int = 1000,
    batch_size: int = 32  # Increased from 15 to 32 for GPU efficiency
):
    """
    Main CLTL processing function
    
    CLTL Strategy: English context + Vietnamese questions/answers
    This preserves international medical knowledge while enabling Vietnamese reasoning
    """
    print("🚀 Starting Cross-Lingual Transfer Learning (CLTL) pipeline")
    print(f"📊 Configuration:")
    print(f"  Translation: NLLB-200 → Helsinki-NLP (best quality for medical Vietnamese)")
    print(f"  Inference: float16 mixed precision with beam search (num_beams=5)")
    print(f"  Batch Size: {batch_size} (GPU-optimized)")
    print(f"  Context: English (preserves international medical knowledge)")
    print(f"  Questions/Answers: Vietnamese (enables local reasoning)")
    print(f"  Max PubMedQA: {max_pubmedqa} (yes/no only, no 'maybe')")
    print(f"  Max BioASQ: {max_bioasq} (yesno questions only)")
    
    # Extract samples with context
    pubmedqa_samples = extract_pubmedqa_cltl_samples.remote(max_pubmedqa)
    bioasq_samples = extract_bioasq_cltl_samples.remote(max_bioasq)
    
    # Combine all samples
    all_samples = pubmedqa_samples + bioasq_samples
    random.seed(42)
    random.shuffle(all_samples)
    
    print(f"📊 Total samples for CLTL: {len(all_samples)}")
    
    # Process in batches
    all_cltl_samples = []
    total_batches = (len(all_samples) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"🔄 Processing CLTL batch {batch_num}/{total_batches}")
        
        # Translate batch for CLTL (English context + Vietnamese Q&A)
        cltl_batch = translate_cltl_batch.remote(batch)
        all_cltl_samples.extend(cltl_batch)
    
    print(f"✅ Completed CLTL translation: {len(all_cltl_samples)} samples")
    
    # Create train/test split
    random.shuffle(all_cltl_samples)
    split_idx = int(0.9 * len(all_cltl_samples))
    
    train_samples = all_cltl_samples[:split_idx]
    test_samples = all_cltl_samples[split_idx:]
    
    # Save CLTL training data
    context_suffix = "_cltl"  # CLTL: English context + Vietnamese Q&A
    train_file = f"medical_qa_vietnamese{context_suffix}_train.jsonl"
    
    with open(f"/data/{train_file}", "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Save CLTL test data
    test_file = f"medical_qa_vietnamese{context_suffix}_test.jsonl"
    
    with open(f"/data/{test_file}", "w", encoding="utf-8") as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Generate CLTL statistics
    stats = {
        "cltl_configuration": {
            "context_language": "english_only",
            "question_language": "vietnamese",
            "answer_language": "vietnamese",
            "knowledge_transfer": "international_to_vietnamese"
        },
        "dataset_stats": {
            "total_samples": len(all_cltl_samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "pubmedqa_source": len(pubmedqa_samples),
            "bioasq_source": len(bioasq_samples)
        },
        "answer_distribution": {
            "Đúng": sum(1 for s in all_cltl_samples if s["output"] == "Đúng"),
            "Sai": sum(1 for s in all_cltl_samples if s["output"] == "Sai")
        },
        "context_preservation": {
            "english_contexts": len([s for s in all_cltl_samples if not s.get("context_bilingual", False)]),
            "bilingual_contexts": len([s for s in all_cltl_samples if s.get("context_bilingual", False)])
        }
    }
    
    stats_file = f"medical_qa_vietnamese{context_suffix}_stats.json"
    with open(f"/data/{stats_file}", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("📊 CLTL Statistics:")
    print(f"  Total samples: {stats['dataset_stats']['total_samples']}")
    print(f"  Training: {stats['dataset_stats']['train_samples']}")
    print(f"  Testing: {stats['dataset_stats']['test_samples']}")
    print(f"  Context mode: {stats['cltl_configuration']['context_language']}")
    print(f"  Đúng: {stats['answer_distribution']['Đúng']}")
    print(f"  Sai: {stats['answer_distribution']['Sai']}")
    
    return {
        "train_file": train_file,
        "test_file": test_file,
        "stats_file": stats_file,
        "stats": stats
    }

@app.local_entrypoint()
def main(
    max_pubmedqa: int = 2000,
    max_bioasq: int = 1000
):
    """
    Cross-Lingual Transfer Learning Translation Pipeline
    
    Args:
        max_pubmedqa: Maximum PubMedQA samples
        max_bioasq: Maximum BioASQ samples  
    
    Usage:
        modal run translate_medical_crosslingual.py
    """
    
    print(f"\n🚀 Starting Cross-Lingual Transfer Learning (CLTL) Pipeline")
    print(f"📋 Strategy: English Context + Vietnamese Q&A")
    
    # Run CLTL processing on Modal
    result = process_cltl_translation.remote(max_pubmedqa, max_bioasq)
    
    print(f"\n🎉 CLTL Translation completed!")
    print(f"\n📥 Download your CLTL datasets:")
    print(f"  modal volume get medical-data {result['train_file']} ./")
    print(f"  modal volume get medical-data {result['test_file']} ./")
    print(f"  modal volume get medical-data {result['stats_file']} ./")
    
    return result

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for sample limits
    max_pubmedqa = 2000
    max_bioasq = 1000
    
    if "--max-pubmedqa" in sys.argv:
        idx = sys.argv.index("--max-pubmedqa")
        if idx + 1 < len(sys.argv):
            max_pubmedqa = int(sys.argv[idx + 1])
    
    if "--max-bioasq" in sys.argv:
        idx = sys.argv.index("--max-bioasq")
        if idx + 1 < len(sys.argv):
            max_bioasq = int(sys.argv[idx + 1])
    
    # Run CLTL pipeline
    main(max_pubmedqa, max_bioasq)