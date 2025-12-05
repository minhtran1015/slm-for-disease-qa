#!/usr/bin/env python3
"""
Simple Modal test for Vietnamese Medical QA model inference
Uses HuggingFace token for base model access
"""

import modal
import json
import os
from typing import List, Dict, Any

# Create Modal app
app = modal.App("vietnamese-medical-qa-simple")

# Create volume for model storage
volume = modal.Volume.from_name("slm_disease_qa-volume", create_if_missing=True)

# Lightweight image with transformers and vLLM
image = modal.Image.from_registry("python:3.11-slim").pip_install([
    "torch",
    "transformers", 
    "peft",
    "accelerate",
    "bitsandbytes",
    "vllm",
    "python-dotenv"
])

@app.cls(
    gpu="H100",
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
class SimpleModelInference:
    """Simple inference class for testing"""
    
    @modal.enter()
    def setup(self):
        """Setup model with HuggingFace token"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import os
        
        print("🔧 Loading HuggingFace token...")
        
        # Load HF token from .env file
        hf_token = None
        try:
            with open('/vol/.env', 'r') as f:
                for line in f:
                    if line.startswith('HF_ACCESS_TOKEN='):
                        hf_token = line.split('=', 1)[1].strip()
                        break
            print("✅ HuggingFace token loaded successfully")
        except FileNotFoundError:
            print("❌ .env file not found")
            return
        
        # Set token in environment
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
        
        # Model paths (using correct base model from adapter config)
        base_model_name = "google/gemma-3-1b-it"  # Correct base model from adapter_config.json
        adapter_path = "/vol/models/gemma-1b-finetuned"
        
        print(f"📦 Loading base model: {base_model_name}")
        print(f"🔗 Loading LoRA adapter: {adapter_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, 
            token=hf_token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    @modal.method()
    def predict(self, prompt: str) -> Dict[str, str]:
        """Generate prediction for a single prompt"""
        import torch
        
        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=10,  # Only need "Đúng" or "Sai"
                do_sample=False, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0.1
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip()
        
        # Classify
        if response.lower().startswith("đúng"):
            classification = "Đúng"
        elif response.lower().startswith("sai"):
            classification = "Sai"
        else:
            classification = "Unknown"
        
        return {
            "classification": classification,
            "raw_response": response,
            "prompt": prompt
        }

@app.local_entrypoint()
def test_single_prediction():
    """Test a single prediction"""
    model = SimpleModelInference()
    
    test_question = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Paracetamol có tác dụng giảm đau."
    
    print(f"🔍 Testing question: {test_question}")
    result = model.predict.remote(test_question)
    
    print(f"📋 Result: {result}")
    print(f"🎯 Classification: {result['classification']}")
    
    return result

@app.local_entrypoint()
def evaluate_test_file(test_file: str = "./train_data/test.jsonl"):
    """Evaluate the test file"""
    print(f"📊 Evaluating test file: {test_file}")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    print(f"📝 Loaded {len(test_data)} test samples")
    
    # Initialize model
    model = SimpleModelInference()
    
    correct = 0
    total = 0
    answered = 0
    
    # Process all samples
    for i, sample in enumerate(test_data):
        user_content = sample['messages'][0]['content']
        expected_answer = sample['messages'][1]['content']
        
        # Get prediction
        result = model.predict.remote(user_content)
        predicted = result['classification']
        
        total += 1
        if predicted in ["Đúng", "Sai"]:
            answered += 1
            if predicted == expected_answer:
                correct += 1
        
        print(f"Sample {i+1}: Expected={expected_answer}, Predicted={predicted} ({'✅' if predicted == expected_answer else '❌'})")
    
    # Calculate metrics
    accuracy = correct / answered if answered > 0 else 0
    answer_rate = answered / total if total > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"Total samples: {total}")
    print(f"Answered: {answered} ({answer_rate:.1%})")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    
    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": accuracy,
        "answer_rate": answer_rate
    }

if __name__ == "__main__":
    print("🚀 Starting simple Modal test...")