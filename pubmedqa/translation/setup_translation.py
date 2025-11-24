#!/usr/bin/env python3
"""
Translation Pipeline Setup Script
==================================
This script installs all dependencies required for the UMLS-aware medical
translation pipeline, including Scispacy, VinAI models, and supporting libraries.

Usage: python setup_translation.py [--gpu|--cpu]
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class SetupManager:
    """Manages the installation process for the translation pipeline."""
    
    SCISPACY_VERSION = "0.5.4"
    SCISPACY_MODEL = "en_core_sci_sm"
    PYTHON_MIN_VERSION = (3, 8)
    
    def __init__(self, mode="auto"):
        """
        Initialize setup manager.
        
        Args:
            mode: Installation mode ('auto', 'gpu', or 'cpu')
        """
        self.mode = mode
        self.install_gpu = False
        
    def print_header(self, text):
        """Print a formatted header."""
        print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}{text}{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")
    
    def print_success(self, text):
        """Print success message."""
        print(f"{Colors.GREEN}✅ {text}{Colors.NC}")
    
    def print_warning(self, text):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.NC}")
    
    def print_error(self, text):
        """Print error message."""
        print(f"{Colors.RED}❌ {text}{Colors.NC}")
    
    def check_python_version(self):
        """Check if Python version meets requirements."""
        self.print_header("Step 1/6: Checking Python Environment")
        
        current_version = sys.version_info
        if current_version < self.PYTHON_MIN_VERSION:
            self.print_error(
                f"Python {self.PYTHON_MIN_VERSION[0]}.{self.PYTHON_MIN_VERSION[1]}+ is required. "
                f"Found {current_version.major}.{current_version.minor}"
            )
            sys.exit(1)
        
        self.print_success(f"Found Python {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    def check_cuda(self):
        """Check if CUDA is available."""
        self.print_header("Step 2/6: Checking GPU Support")
        
        if self.mode == "cpu":
            self.print_warning("CPU-only mode selected")
            self.install_gpu = False
            return
        
        try:
            # Try to run nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                self.print_success("CUDA detected")
                print(result.stdout.strip())
                self.install_gpu = True
            else:
                self.print_warning("No CUDA detected. Will install CPU-only versions.")
                self.install_gpu = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.print_warning("No CUDA detected. Will install CPU-only versions.")
            self.install_gpu = False
    
    def setup_virtual_environment(self):
        """Optionally create virtual environment."""
        self.print_header("Step 3/6: Setting up Virtual Environment (Optional)")
        
        if Path("venv").exists():
            self.print_warning("Virtual environment already exists at ./venv")
            self.print_warning("Activate with: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)")
            return
        
        response = input("Create virtual environment? (y/n) [recommended]: ").lower()
        
        if response == 'y':
            try:
                subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
                self.print_success("Virtual environment created at ./venv")
                self.print_warning("Please activate it and re-run this script:")
                if platform.system() == "Windows":
                    print("  venv\\Scripts\\activate")
                else:
                    print("  source venv/bin/activate")
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                self.print_error(f"Failed to create virtual environment: {e}")
                sys.exit(1)
        else:
            self.print_warning("Skipping virtual environment creation")
    
    def install_core_dependencies(self):
        """Install core Python packages."""
        self.print_header("Step 4/6: Installing Core Dependencies")
        
        # Upgrade pip
        self.print_success("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        
        # Install PyTorch
        self.print_success("Installing PyTorch...")
        if self.install_gpu:
            # GPU version (CUDA 11.8)
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
        else:
            # CPU version
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True)
        
        # Install other dependencies
        self.print_success("Installing Transformers and supporting libraries...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "transformers", "datasets", "scipy", "tqdm"
        ], check=True)
    
    def install_scispacy(self):
        """Install Scispacy and medical NLP tools."""
        self.print_header("Step 5/6: Installing Scispacy and Medical NLP Tools")
        
        self.print_success("Installing Spacy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
        
        self.print_success("Installing Scispacy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "scispacy"], check=True)
        
        self.print_success(f"Downloading Scispacy model ({self.SCISPACY_MODEL})...")
        model_url = (
            f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
            f"v{self.SCISPACY_VERSION}/{self.SCISPACY_MODEL}-{self.SCISPACY_VERSION}.tar.gz"
        )
        subprocess.run([sys.executable, "-m", "pip", "install", model_url], check=True)
    
    def configure_umls(self):
        """Configure UMLS entity linker."""
        self.print_header("Step 6/6: Configuring UMLS Entity Linker")
        
        self.print_warning("The UMLS linker will download ~2GB of data on first use.")
        self.print_warning("This is automatic and will happen when you run the pipeline.")
    
    def verify_installation(self):
        """Verify that all components are installed correctly."""
        self.print_header("Verifying Installation")
        
        # Test imports
        tests = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("spacy", "Spacy"),
            ("scispacy", "Scispacy"),
        ]
        
        all_passed = True
        for module, name in tests:
            try:
                __import__(module)
                self.print_success(f"{name}: OK")
            except ImportError as e:
                self.print_error(f"{name}: FAILED - {e}")
                all_passed = False
        
        # Test Scispacy model
        try:
            import spacy
            nlp = spacy.load(self.SCISPACY_MODEL)
            self.print_success(f"Scispacy Model ({self.SCISPACY_MODEL}): OK")
        except Exception as e:
            self.print_error(f"Scispacy Model: FAILED - {e}")
            all_passed = False
        
        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.print_success(f"CUDA Support: Available (Device: {device_name})")
            else:
                self.print_warning("CUDA Support: Not available (CPU mode)")
        except Exception as e:
            self.print_error(f"CUDA Check: FAILED - {e}")
        
        if not all_passed:
            sys.exit(1)
    
    def download_vinai_model(self):
        """Optionally pre-download VinAI model."""
        response = input("\nPre-download VinAI translation model (~1.5GB)? (y/n): ").lower()
        
        if response == 'y':
            self.print_header("Downloading VinAI Model")
            
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                
                cache_dir = "./models_cache"
                os.makedirs(cache_dir, exist_ok=True)
                
                print("Downloading VinAI model (this may take a few minutes)...")
                model_name = "vinai/vinai-translate-en2vi"
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", cache_dir=cache_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
                
                self.print_success(f"VinAI model downloaded to {cache_dir}/")
            except Exception as e:
                self.print_error(f"Failed to download VinAI model: {e}")
        else:
            self.print_warning("Model will be downloaded on first use")
    
    def print_next_steps(self):
        """Print next steps after successful installation."""
        self.print_header("✅ Installation Complete!")
        
        print(f"\n{Colors.GREEN}Next Steps:{Colors.NC}")
        print("1. Review configuration: translation_config.json")
        print("2. Run test translation: python test_pipeline.py")
        print("3. Run full translation: python translate_pipeline.py --input ../data/pqaa_train_set.json")
        print(f"\n{Colors.BLUE}For help, see: README.md{Colors.NC}\n")
    
    def run(self):
        """Run the complete setup process."""
        self.print_header("UMLS-Aware Translation Pipeline Setup")
        
        try:
            self.check_python_version()
            self.check_cuda()
            self.setup_virtual_environment()
            self.install_core_dependencies()
            self.install_scispacy()
            self.configure_umls()
            self.verify_installation()
            self.download_vinai_model()
            self.print_next_steps()
            
            self.print_success("Setup complete! 🎉")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Installation failed: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.print_warning("\nSetup interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup script for UMLS-aware translation pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Installation mode (auto=detect, gpu=force GPU, cpu=force CPU)"
    )
    
    args = parser.parse_args()
    
    # Map old style arguments
    mode = args.mode
    if "--gpu" in sys.argv:
        mode = "gpu"
    elif "--cpu" in sys.argv:
        mode = "cpu"
    
    setup = SetupManager(mode=mode)
    setup.run()


if __name__ == "__main__":
    main()
