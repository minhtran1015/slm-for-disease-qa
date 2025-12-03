#!/usr/bin/env python3
"""
Convert ViMedAQA v2 format (chat messages) to standardized format.
"""

import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_v2_to_standardized(input_file: str, output_file: str):
    """
    Convert v2 format to standardized format.
    
    v2 format has:
    - messages: [system, user, assistant]
    - answer, answer_vi, question, question_type, statement_id, source
    
    Standardized format has:
    - instruction: "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: " + question
    - input: ""
    - output: "Đúng" or "Sai"
    - question_type, statement_id, source (preserved)
    """
    
    standardized_prefix = "Dựa trên kiến thức y khoa, hãy xác minh thông tin sau là Đúng hay Sai: "
    
    converted_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # Parse the JSON line
                    data = json.loads(line.strip())
                    
                    # Extract the question from the user message
                    question = None
                    output = None
                    
                    # Get question from messages
                    if 'messages' in data:
                        for message in data['messages']:
                            if message['role'] == 'user':
                                question = message['content']
                            elif message['role'] == 'assistant':
                                output = message['content']
                    
                    # Fallback to 'question' field if not found in messages
                    if not question and 'question' in data:
                        question = data['question']
                    
                    # Fallback to answer_vi for output if not found in messages
                    if not output and 'answer_vi' in data:
                        output = "Đúng" if data['answer_vi'].lower() == "đúng" else "Sai"
                    
                    # Validate required fields
                    if not question:
                        logger.warning(f"Line {line_num}: No question found, skipping")
                        continue
                    
                    if not output:
                        logger.warning(f"Line {line_num}: No output found, skipping")
                        continue
                    
                    # Create standardized format
                    standardized_data = {
                        "instruction": standardized_prefix + question,
                        "input": "",
                        "output": output,
                        "question_type": data.get("question_type", "unknown"),
                        "statement_id": data.get("statement_id", "unknown"),
                        "source": data.get("source", "vimedaqa")
                    }
                    
                    # Write to output file
                    outfile.write(json.dumps(standardized_data, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                    if converted_count % 1000 == 0:
                        logger.info(f"Converted {converted_count} samples...")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode error - {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error - {e}")
                    continue
    
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return False
    except Exception as e:
        logger.error(f"File processing error: {e}")
        return False
    
    logger.info(f"Conversion completed! Converted {converted_count} samples to {output_file}")
    return True

def main():
    input_file = "vimedaqa_yesno_train_v2.jsonl"
    output_file = "vimedaqa_yesno_train_v2_standardized.jsonl"
    
    logger.info(f"Converting {input_file} to standardized format...")
    success = convert_v2_to_standardized(input_file, output_file)
    
    if success:
        logger.info("Conversion successful!")
        
        # Validate the output
        logger.info("Validating converted data...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                sample_count = 0
                for line in f:
                    data = json.loads(line.strip())
                    sample_count += 1
                    if sample_count <= 3:
                        logger.info(f"Sample {sample_count}: {json.dumps(data, ensure_ascii=False)[:200]}...")
                
                logger.info(f"Total samples in output file: {sample_count}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
    else:
        logger.error("Conversion failed!")

if __name__ == "__main__":
    main()