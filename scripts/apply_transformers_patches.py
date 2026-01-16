#!/usr/bin/env python3
"""
Apply transformers patches required for PI0.5 PyTorch implementation.

This script copies the patched transformers files from OpenPI into your
installed transformers package. These patches are required for:
1. Supporting AdaRMS normalization
2. Correctly controlling activation precision
3. Allowing KV cache to be used without being updated

Run this script after installing transformers==4.53.2.
"""

import shutil
import site
import sys
from pathlib import Path


def main():
    # Find transformers installation
    site_packages = Path(site.getsitepackages()[0])
    transformers_path = site_packages / "transformers"
    
    if not transformers_path.exists():
        print(f"❌ Error: transformers package not found at {transformers_path}")
        print("Please install transformers==4.53.2 first:")
        print("  pip install transformers==4.53.2")
        sys.exit(1)
    
    # Find patches directory
    script_dir = Path(__file__).parent.parent
    patches_dir = script_dir / "third_party" / "transformers_patches"
    
    if not patches_dir.exists():
        print(f"❌ Error: patches directory not found at {patches_dir}")
        sys.exit(1)
    
    # Apply patches
    print(f"📦 Applying transformers patches...")
    print(f"   Source: {patches_dir}")
    print(f"   Target: {transformers_path}")
    
    # Copy all files from patches_dir to transformers_path
    files_copied = 0
    for item in patches_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(patches_dir)
            target_file = transformers_path / relative_path
            
            # Create parent directory if needed
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, target_file)
            files_copied += 1
            print(f"   ✓ {relative_path}")
    
    print(f"\n✅ Successfully applied {files_copied} patches to transformers")
    print("\n⚠️  WARNING: These patches permanently modify your transformers installation.")
    print("   To revert, reinstall transformers: pip install --force-reinstall transformers==4.53.2")


if __name__ == "__main__":
    main()
