#!/usr/bin/env python3
"""
Modal.com High-Performance Inference for Fine-tuned Gemma-1B Vietnamese Medical Model
Goal: Maximize inference throughput (Speed > Cost)
"""

import modal
import time
from typing import List, Dict, Any
import json

# Modal configuration
app = modal.App("slm-disease-qa-inference")
volume = modal.Volume.from_name("slm_disease_qa-volume", create_if_missing=True)

# High-performance image with vLLM
image = modal.Image.from_registry("vllm/vllm-openai:latest").pip_install(
    "transformers",
    "torch",
    "peft",
    "accelerate",
    "bitsandbytes",
    "python-dotenv"
)

@app.cls(
    gpu="H100",  # Maximum performance GPU
    image=image,
    volumes={"/vol": volume},
    concurrency_limit=100,  # High concurrency for batching
    timeout=600,  # 10 minutes timeout
)
class ModelInference:
    """High-performance inference class for Vietnamese Medical Gemma-1B model"""
    
    @modal.enter()
    def setup(self):
        """Initialize vLLM engine with maximum performance settings"""
        import torch
        import os
        from vllm import AsyncLLMEngine, LLM, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        print("🚀 Initializing vLLM engine for LoRA adapter...")
        
        # Get HuggingFace token from environment file
        hf_token = None
        try:
            with open('/vol/.env', 'r') as f:
                for line in f:
                    if line.startswith('HF_ACCESS_TOKEN='):
                        hf_token = line.split('=', 1)[1].strip()
                        break
        except FileNotFoundError:
            print("⚠️ Warning: .env file not found, proceeding without HF token")
        
        # Set HF token in environment for HuggingFace model downloads
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            print("✅ HuggingFace token loaded from .env")
        
        # Model paths
        base_model_name = "google/gemma-1b-it"  # Base model
        lora_path = "/vol/models/gemma-1b-finetuned"  # LoRA adapter path
        
        # vLLM configuration with LoRA support
        engine_args = {
            "model": base_model_name,  # Use base model
            "enable_lora": True,  # Enable LoRA support
            "lora_modules": [("gemma_medical", lora_path)],  # Load LoRA adapter
            "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
            "max_model_len": 512,  # Reasonable context length for medical QA
            "dtype": "float16",  # Fast inference
            "trust_remote_code": True,
        }
        
        try:
            # Initialize synchronous LLM engine for better throughput
            self.llm = LLM(**engine_args)
            print(f"✅ vLLM engine initialized successfully")
            print(f"📊 Base Model: {base_model_name}")
            print(f"📎 LoRA Adapter: {lora_path}")
            print(f"🔥 GPU Memory Utilization: 90%")
            print(f"⚡ CUDA Graphs: Enabled")
            print(f"🎯 Max Concurrent Sequences: 256")
            
            # Pre-configure sampling parameters for medical QA
            self.sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for consistent medical answers
                max_tokens=512,   # Sufficient for medical responses
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop_token_ids=[],
                skip_special_tokens=True
            )
            
            # Warm up the model with a dummy request
            warmup_prompt = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Paracetamol có phải là thuốc giảm đau không?"
            _ = self.llm.generate([warmup_prompt], self.sampling_params)
            print("🔥 Model warmed up successfully")
            
        except Exception as e:
            print(f"❌ Error initializing vLLM: {e}")
            raise
    
    @modal.method()
    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of prompts with maximum throughput
        
        Args:
            prompts: List of input prompts for medical QA
            
        Returns:
            List of generation results with timing info
        """
        if not prompts:
            return []
        
        start_time = time.time()
        batch_size = len(prompts)
        
        print(f"🔄 Processing batch of {batch_size} prompts...")
        
        try:
            # Generate responses using vLLM batch processing
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Process results
            results = []
            total_tokens = 0
            
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                total_tokens += completion_tokens
                
                # Extract medical classification (Đúng/Sai/Unknown)
                classification = "Unknown"
                if generated_text.lower().startswith("đúng"):
                    classification = "Đúng"
                elif generated_text.lower().startswith("sai"):
                    classification = "Sai"
                
                results.append({
                    "prompt_index": i,
                    "prompt": prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i],
                    "generated_text": generated_text,
                    "classification": classification,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                })
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate performance metrics
            tokens_per_second = total_tokens / duration if duration > 0 else 0
            prompts_per_second = batch_size / duration if duration > 0 else 0
            
            print(f"✅ Batch completed in {duration:.2f}s")
            print(f"📊 Performance: {tokens_per_second:.1f} tokens/sec, {prompts_per_second:.1f} prompts/sec")
            print(f"🎯 Total tokens generated: {total_tokens}")
            
            # Add batch-level metadata
            batch_metadata = {
                "batch_size": batch_size,
                "duration_seconds": duration,
                "tokens_per_second": tokens_per_second,
                "prompts_per_second": prompts_per_second,
                "total_tokens_generated": total_tokens
            }
            
            return {
                "results": results,
                "metadata": batch_metadata
            }
            
        except Exception as e:
            print(f"❌ Error during batch generation: {e}")
            raise

    @modal.method()
    def generate_single(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response for a single prompt (convenience method)
        
        Args:
            prompt: Single input prompt for medical QA
            
        Returns:
            Generation result
        """
        batch_result = self.generate_batch([prompt])
        if batch_result["results"]:
            return batch_result["results"][0]
        return {"error": "No result generated"}

    @modal.method()
    def evaluate_test_file(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate model on test data in the format of test.jsonl
        
        Args:
            test_data: List of test samples with messages format
            
        Returns:
            Evaluation results with accuracy metrics
        """
        if not test_data:
            return {"error": "No test data provided"}
        
        print(f"📊 Starting evaluation on {len(test_data)} test samples...")
        
        # Extract prompts and expected answers
        prompts = []
        expected_answers = []
        
        for item in test_data:
            if "messages" in item and len(item["messages"]) >= 2:
                user_message = item["messages"][0]["content"]
                expected_answer = item["messages"][1]["content"]
                
                prompts.append(user_message)
                expected_answers.append(expected_answer)
        
        if not prompts:
            return {"error": "No valid test samples found"}
        
        # Run batch inference
        batch_result = self.generate_batch(prompts)
        results = batch_result["results"]
        
        # Calculate accuracy metrics
        total_samples = len(results)
        correct_predictions = 0
        answered_samples = 0
        
        detailed_results = []
        
        for i, result in enumerate(results):
            expected = expected_answers[i]
            predicted = result["classification"]
            
            is_answered = predicted != "Unknown"
            is_correct = predicted == expected
            
            if is_answered:
                answered_samples += 1
                if is_correct:
                    correct_predictions += 1
            
            detailed_results.append({
                "sample_id": i + 1,
                "prompt": result["prompt"],
                "expected": expected,
                "predicted": predicted,
                "generated_text": result["generated_text"],
                "is_correct": is_correct,
                "is_answered": is_answered
            })
        
        # Calculate metrics
        answer_rate = answered_samples / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / answered_samples if answered_samples > 0 else 0
        
        # Performance classification
        if accuracy >= 0.75:
            performance_level = "🏆 EXCELLENT"
        elif accuracy >= 0.65:
            performance_level = "🥇 VERY GOOD"
        elif accuracy >= 0.55:
            performance_level = "🥈 GOOD"
        else:
            performance_level = "⚠️ NEEDS IMPROVEMENT"
        
        evaluation_summary = {
            "total_samples": total_samples,
            "answered_samples": answered_samples,
            "correct_predictions": correct_predictions,
            "answer_rate": answer_rate,
            "accuracy": accuracy,
            "performance_level": performance_level,
            "batch_metadata": batch_result["metadata"]
        }
        
        print(f"✅ Evaluation completed:")
        print(f"📊 Total: {total_samples}, Answered: {answered_samples}, Correct: {correct_predictions}")
        print(f"🎯 Answer Rate: {answer_rate:.1%}, Accuracy: {accuracy:.1%}")
        print(f"🏆 Performance: {performance_level}")
        
        return {
            "summary": evaluation_summary,
            "detailed_results": detailed_results
        }

    @modal.method()
    def health_check(self) -> Dict[str, str]:
        """Health check endpoint for monitoring"""
        try:
            # Simple test generation
            test_prompt = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Aspirin có tác dụng chống viêm không?"
            result = self.generate_single(test_prompt)
            
            return {
                "status": "healthy",
                "model_path": "/vol/models/gemma-1b-finetuned",
                "test_classification": result.get("classification", "Unknown"),
                "timestamp": str(time.time())
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": str(time.time())
            }

@app.local_entrypoint()
def local_entrypoint():
    """
    Local test entrypoint to measure inference speed
    """
    print("🧪 Starting high-performance inference speed test...")
    
    # Create test batch of Vietnamese medical questions
    test_prompts = [
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Paracetamol có phải là thuốc giảm đau không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Aspirin có tác dụng chống viêm không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Amoxicillin là thuốc kháng sinh không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Suy tim có phải là bệnh lý tim mạch không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Đau đầu là triệu chứng của nhiều bệnh khác nhau không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Insulin được sử dụng điều trị tiểu đường không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Huyết áp cao có thể gây đột quỵ không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Vitamin C giúp tăng cường hệ miễn dịch không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Cholesterol cao có thể gây bệnh tim mạch không?",
        "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: Thuốc lá có thể gây ung thư phổi không?"
    ]
    
    # Scale up to 100 prompts by repeating
    large_batch = (test_prompts * 10)  # 100 prompts total
    
    print(f"📦 Prepared batch of {len(large_batch)} prompts")
    print("🚀 Starting batch inference...")
    
    # Measure total time including Modal overhead
    total_start = time.time()
    
    # Run inference
    model = ModelInference()
    
    # Health check first
    health = model.health_check.remote()
    print(f"🩺 Health check: {health.get('status', 'unknown')}")
    
    # Main batch inference
    result = model.generate_batch.remote(large_batch)
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    # Display results
    print("\n" + "="*60)
    print("📊 PERFORMANCE RESULTS")
    print("="*60)
    
    metadata = result.get("metadata", {})
    
    print(f"Batch Size: {metadata.get('batch_size', 0)} prompts")
    print(f"Processing Time: {metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"Total Time (with overhead): {total_duration:.2f} seconds")
    print(f"Tokens per Second: {metadata.get('tokens_per_second', 0):.1f} TPS")
    print(f"Prompts per Second: {metadata.get('prompts_per_second', 0):.1f} PPS")
    print(f"Total Tokens Generated: {metadata.get('total_tokens_generated', 0)}")
    
    # Show sample results
    print(f"\n🎯 Sample Results:")
    results = result.get("results", [])
    for i, res in enumerate(results[:3]):  # Show first 3
        print(f"  {i+1}. {res.get('classification', 'Unknown')} - {res.get('generated_text', '')[:50]}...")
    
    print(f"\n🏆 H100 + vLLM Performance: {metadata.get('tokens_per_second', 0):.0f} TPS")
    print("✅ High-performance inference test completed!")

# Additional helper functions for model management

@app.function(
    image=modal.Image.from_registry("python:3.11").pip_install(
        "huggingface_hub", "transformers"
    ),
    volumes={"/vol": volume},
    timeout=1800  # 30 minutes for model upload
)
def upload_model_to_volume():
    """
    Upload the fine-tuned model from local checkpoint to Modal volume
    Run this once to prepare the volume
    """
    import shutil
    import os
    
    # Model path relative to current directory
    local_model_path = "./checkpoint+eval_script/gemma-1b-vietnamese-medical-v2-100k"
    volume_model_path = "/vol/models/gemma-1b-finetuned"
    
    print(f"📁 Uploading model from {local_model_path} to {volume_model_path}")
    
    # Create directory structure
    os.makedirs(os.path.dirname(volume_model_path), exist_ok=True)
    
    # Copy model files
    if os.path.exists(local_model_path):
        shutil.copytree(local_model_path, volume_model_path, dirs_exist_ok=True)
        print(f"✅ Model uploaded successfully to volume")
        
        # List files to verify
        files = os.listdir(volume_model_path)
        print(f"📋 Files in volume: {files}")
    else:
        print(f"❌ Local model path not found: {local_model_path}")
        raise FileNotFoundError(f"Model not found at {local_model_path}")

@app.local_entrypoint()
def evaluate_test_jsonl(test_file: str = "./train_data/test.jsonl"):
    """
    Convenience entrypoint for evaluating test.jsonl files
    
    Args:
        test_file: Path to test.jsonl file
    """
    print(f"📊 Starting evaluation mode with file: {test_file}")
    
    # Load test data
    try:
        import json
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        print(f"📁 Loaded {len(test_data)} test samples")
        
        # Initialize model and run evaluation
        model = ModelInference()
        
        # Health check first
        health = model.health_check.remote()
        print(f"🩺 Health check: {health['status']}")
        
        # Run evaluation
        eval_result = model.evaluate_test_file.remote(test_data)
        
        # Display evaluation results
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        summary = eval_result["summary"]
        
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Answered Samples: {summary['answered_samples']}")
        print(f"Correct Predictions: {summary['correct_predictions']}")
        print(f"Answer Rate: {summary['answer_rate']:.1%}")
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Performance Level: {summary['performance_level']}")
        
        # Performance metrics
        batch_meta = summary['batch_metadata']
        print(f"\n⚡ Performance Metrics:")
        print(f"Tokens per Second: {batch_meta['tokens_per_second']:.1f} TPS")
        print(f"Processing Time: {batch_meta['duration_seconds']:.2f} seconds")
        
        # Show some sample incorrect predictions
        detailed = eval_result["detailed_results"]
        incorrect_samples = [r for r in detailed if r['is_answered'] and not r['is_correct']]
        
        if incorrect_samples:
            print(f"\n❌ Sample Incorrect Predictions (first 3):")
            for i, sample in enumerate(incorrect_samples[:3]):
                print(f"  {i+1}. Expected: {sample['expected']}, Got: {sample['predicted']}")
                print(f"     Question: {sample['prompt'][:100]}...")
                print(f"     Generated: {sample['generated_text'][:50]}...\n")
        
        print("✅ Evaluation completed!")
        
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        raise

if __name__ == "__main__":
    print("🔧 Modal SLM Disease QA Inference")
    print("Available commands:")
    print("  modal run inference.py::upload_model_to_volume          # Upload model to volume")
    print("  modal run inference.py::local_entrypoint                # Speed test (default)")
    print("  modal run inference.py::evaluate_test_jsonl             # Evaluate test.jsonl")
    print("  modal run inference.py::evaluate_test_jsonl --test-file path/to/test.jsonl  # Evaluate custom file")
    print("")
    print("Examples:")
    print("  # Upload model (run once)")
    print("  modal run inference.py::upload_model_to_volume")
    print("")
    print("  # Speed test")
    print("  modal run inference.py::local_entrypoint")
    print("")
    print("  # Evaluate model on test.jsonl")
    print("  modal run inference.py::evaluate_test_jsonl")
    print("")
    print("  # Evaluate on custom file")
    print("  modal run inference.py::evaluate_test_jsonl --test-file ./train_data/val.jsonl")