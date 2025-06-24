#!/usr/bin/env python3
"""
Quick installation test for WEAVER
Run this to verify everything is working after installation.
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, description="", optional=False):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except ImportError as e:
        print(f"{'❌' if not optional else '⚠️'} {module_name} {description} - Error: {e}")
        return False

def main():
    print("🧪 Testing WEAVER installation...\n")
    
    # Test core dependencies
    core_deps = [
        ("torch", "- PyTorch"),
        ("transformers", "- Hugging Face Transformers"),
        ("datasets", "- Hugging Face Datasets"),
        ("hydra", "- Hydra configuration"),
        ("omegaconf", "- OmegaConf"),
        ("sklearn", "- Scikit-learn"),
        ("pandas", "- Pandas"),
        ("numpy", "- NumPy"),
        ("metal", "- Metal-AMA (required for weak supervision)"),
        ("vllm", "- vLLM inference"),
        ("sentence_transformers", "- Sentence Transformers"),
    ]
    
    print("Core dependencies:")
    failed_core = []
    for module, desc in core_deps:
        if not test_import(module, desc):
            failed_core.append(module)
    
    print("\nWEAVER modules:")
    weaver_modules = [
        ("weaver", "- Core package"),
        ("weaver.models", "- Model classes"),
        ("weaver.dataset", "- Dataset handling"),
    ]
    
    failed_weaver = []
    for module, desc in weaver_modules:
        if not test_import(module, desc):
            failed_weaver.append(module)
    
    print("\nOptional dependencies:")
    optional_deps = [
        ("wandb", "- Weights & Biases logging"),
        ("flash_attn", "- FlashAttention (GPU acceleration)"),
    ]
    
    for module, desc in optional_deps:
        test_import(module, desc, optional=True)
    
    # Test that we can instantiate key classes
    print("\nTesting class instantiation:")
    try:
        from weaver.dataset import VerificationDataset
        print("✓ Can import VerificationDataset")
    except Exception as e:
        print(f"❌ VerificationDataset import failed: {e}")
        failed_weaver.append("VerificationDataset")
    
    try:
        from weaver.models import Model
        print("✓ Can import Model")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        failed_weaver.append("Model")
    
    # Summary
    print("\n" + "="*50)
    if not failed_core and not failed_weaver:
        print("🎉 All tests passed! WEAVER is ready to use. Continue following README.md for further instructions!")
        return 0
    else:
        print("❌ Some tests failed:")
        if failed_core:
            print(f"   Core dependencies: {', '.join(failed_core)}")
        if failed_weaver:
            print(f"   WEAVER modules: {', '.join(failed_weaver)}")
        print("\nTry running:")
        print("  pip install -e .")
        print("  cd metal-ama && pip install -e .")
        return 1

if __name__ == "__main__":
    sys.exit(main())