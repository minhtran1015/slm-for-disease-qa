"""
VinAI Translator Utility Module
================================
Standalone module for Vietnamese translation using VinAI models.
Can be used independently of the main pipeline for general translation tasks.

Usage:
    from translator import VinAITranslator
    
    translator = VinAITranslator()
    result = translator.translate("Heart attack symptoms include chest pain")
    print(result)  # "Triệu chứng cơn đau tim bao gồm đau ngực"
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class VinAITranslator:
    """
    Vietnamese translator using VinAI's pretrained models.
    Supports batch translation with GPU acceleration.
    """
    
    def __init__(
        self,
        model_name: str = "vinai/vinai-translate-en2vi",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        num_beams: int = 4,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the VinAI translator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading VinAI model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang="en_XX",
            cache_dir=cache_dir
        )
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ VinAI Translator ready ({self.device})")
    
    def translate(self, text: str) -> str:
        """
        Translate a single text.
        
        Args:
            text: English text to translate
            
        Returns:
            Vietnamese translation
        """
        return self.translate_batch([text])[0]
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple texts in batch.
        
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
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                decoder_start_token_id=self.tokenizer.lang_code_to_id["vi_VN"],
                num_return_sequences=1,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return translations
    
    def translate_large_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a large list of texts by processing in chunks.
        
        Args:
            texts: List of English texts
            
        Returns:
            List of Vietnamese translations
        """
        all_translations = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            translations = self.translate_batch(batch)
            all_translations.extend(translations)
        
        return all_translations
    
    def translate_with_alternatives(
        self,
        text: str,
        num_alternatives: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple translation alternatives using beam search.
        
        Args:
            text: English text to translate
            num_alternatives: Number of alternative translations to generate
            
        Returns:
            List of translation dictionaries with scores
        """
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with multiple returns
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                decoder_start_token_id=self.tokenizer.lang_code_to_id["vi_VN"],
                num_return_sequences=num_alternatives,
                num_beams=max(num_alternatives, self.num_beams),
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Decode all alternatives
        translations = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        # Format results
        results = []
        for i, translation in enumerate(translations):
            results.append({
                "rank": i + 1,
                "translation": translation,
                "text": translation  # Alias for compatibility
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "vocab_size": self.tokenizer.vocab_size,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }


class MedicalTermTranslator(VinAITranslator):
    """
    Specialized translator for medical terms with pre/post-processing.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Medical term mappings (optional enhancements)
        self.medical_mappings = {
            "myocardial infarction": "nhồi máu cơ tim",
            "atrial fibrillation": "rung nhĩ",
            "hypertension": "tăng huyết áp",
            "diabetes mellitus": "đái tháo đường",
            "stroke": "đột quỵ",
            # Add more as needed
        }
    
    def translate_medical(self, text: str, use_mappings: bool = True) -> str:
        """
        Translate medical text with optional term replacement.
        
        Args:
            text: Medical text in English
            use_mappings: Whether to use predefined medical term mappings
            
        Returns:
            Vietnamese translation
        """
        if use_mappings:
            # Pre-process: Standardize medical terms
            text_lower = text.lower()
            for eng_term, vi_term in self.medical_mappings.items():
                if eng_term in text_lower:
                    # Mark for post-processing
                    text = text.replace(eng_term, f"[MEDTERM:{eng_term}]")
        
        # Translate
        translation = self.translate(text)
        
        if use_mappings:
            # Post-process: Replace markers with correct Vietnamese terms
            for eng_term, vi_term in self.medical_mappings.items():
                translation = translation.replace(f"[MEDTERM:{eng_term}]", vi_term)
        
        return translation


def main():
    """Demo usage of VinAI Translator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VinAI Vietnamese Translator")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--file", type=str, help="File containing text (one per line)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--alternatives", type=int, help="Generate N alternative translations")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = VinAITranslator(
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Print model info
    info = translator.get_model_info()
    print(f"\n{'='*60}")
    print("VinAI Translator Information")
    print(f"{'='*60}")
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Parameters: {info['model_parameters']:,}")
    print(f"{'='*60}\n")
    
    # Get input text
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [
            "Heart attack symptoms include chest pain and shortness of breath.",
            "Diabetes is a chronic disease that affects blood sugar levels.",
            "Regular exercise can help prevent cardiovascular disease."
        ]
        print("Using demo texts...\n")
    
    # Translate
    print(f"Translating {len(texts)} text(s)...\n")
    
    if args.alternatives:
        # Generate alternatives for first text
        print("Generating alternative translations...\n")
        alternatives = translator.translate_with_alternatives(texts[0], args.alternatives)
        
        print(f"Original: {texts[0]}\n")
        for alt in alternatives:
            print(f"{alt['rank']}. {alt['translation']}")
    else:
        # Standard translation
        if len(texts) == 1:
            translation = translator.translate(texts[0])
            print(f"Original: {texts[0]}")
            print(f"Translation: {translation}")
        else:
            translations = translator.translate_large_batch(texts)
            
            for i, (original, translation) in enumerate(zip(texts, translations), 1):
                print(f"{i}. Original: {original}")
                print(f"   Translation: {translation}\n")


if __name__ == "__main__":
    main()
