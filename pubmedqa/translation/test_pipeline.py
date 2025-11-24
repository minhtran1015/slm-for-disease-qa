#!/usr/bin/env python3
"""
Quick Test Script for Translation Pipeline
===========================================
This script runs a quick test with 10 samples to verify the pipeline works.

Usage: python test_pipeline.py
"""

import sys
import os
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.NC}")


def check_dependencies():
    """Check if dependencies are installed."""
    print_header("🧪 Testing UMLS Translation Pipeline")
    
    try:
        import spacy
        import scispacy
        import transformers
        print_success("Dependencies found")
        return True
    except ImportError as e:
        print_error("Dependencies not installed!")
        print(f"Missing: {e.name}")
        print("\nPlease run: python setup_translation.py")
        return False


def run_test():
    """Run the translation pipeline test."""
    # Define paths
    input_file = Path("../data/pqaa_train_set.json")
    output_file = Path("./outputs/test_translation.json")
    config_file = Path("./translation_config.json")
    
    # Check if input file exists
    if not input_file.exists():
        print_error(f"Input file not found: {input_file}")
        print("\nPlease ensure PubMedQA data is in the correct location.")
        return False
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("Running translation test (10 samples)...\n")
    
    # Build command
    cmd = [
        sys.executable,
        "translate_pipeline.py",
        "--input", str(input_file),
        "--output", str(output_file),
        "--max-samples", "10",
        "--config", str(config_file)
    ]
    
    # Run pipeline
    try:
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print_header("✅ Test completed successfully!")
            
            print(f"Output saved to: {output_file}\n")
            print("Next steps:")
            print(f"1. View output: python -m json.tool {output_file} | head -n 50")
            print("2. Run full dataset: python translate_pipeline.py --input ../data/pqaa_train_set.json")
            print("")
            
            return True
        else:
            print_error("Test failed. Please check the error messages above.")
            return False
            
    except subprocess.CalledProcessError as e:
        print_error(f"Test failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.NC}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False


def main():
    """Main entry point."""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run test
    success = run_test()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
