#!/usr/bin/env python3
"""
MOBPY Release Script for PyPI
Version 2.0.0

This script automates the release process for MOBPY to PyPI.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    
    return result


def clean_build_dirs():
    """Remove previous build artifacts."""
    print("🧹 Cleaning previous builds...")
    dirs_to_clean = ["dist", "build", "src/MOBPY.egg-info"]
    
    for dir_path in dirs_to_clean:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_path}")


def main():
    print("🚀 MOBPY Release Script v2.0.0")
    print("=" * 40)
    
    # Check Python version
    print("\n📌 Checking Python version...")
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    # Clean previous builds
    clean_build_dirs()
    
    # Upgrade build tools
    print("\n⬆️ Upgrading build tools...")
    run_command("pip install --upgrade pip setuptools wheel twine build")
    
    # Run tests
    print("\n🧪 Running tests...")
    test_result = run_command("pytest tests/ -v --tb=short", check=False)
    
    if test_result.returncode != 0:
        print("❌ Tests failed. Please fix before releasing.")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            sys.exit(1)
    else:
        print("✅ All tests passed!")
    
    # Build the package
    print("\n📦 Building package...")
    run_command("python -m build")
    
    # Check the package
    print("\n🔍 Checking package with twine...")
    run_command("twine check dist/*")
    
    # Display package contents
    print("\n📋 Package contents:")
    for file in Path("dist").glob("*"):
        size_kb = file.stat().st_size / 1024
        print(f"   {file.name} ({size_kb:.1f} KB)")
    
    # Ask for upload confirmation
    print("\n⚠️  Ready to upload to PyPI?")
    print("   Package: MOBPY")
    print("   Version: 2.0.0")
    print()
    
    test_upload = input("Upload to TestPyPI first? (recommended) [y/N]: ").strip().lower()
    
    if test_upload == 'y':
        print("\n📤 Uploading to TestPyPI...")
        run_command("twine upload --repository testpypi dist/*")
        
        print("\n✅ Uploaded to TestPyPI!")
        print("   Test install with:")
        print("   pip install --index-url https://test.pypi.org/simple/ MOBPY")
        print()
        
        prod_upload = input("Continue to production PyPI? [y/N]: ").strip().lower()
        
        if prod_upload == 'y':
            print("\n📤 Uploading to PyPI...")
            run_command("twine upload dist/*")
            print("✅ Successfully uploaded MOBPY 2.0.0 to PyPI!")
        else:
            print("⏸️  Production upload cancelled.")
    else:
        direct_upload = input("Upload directly to PyPI? [y/N]: ").strip().lower()
        
        if direct_upload == 'y':
            print("\n📤 Uploading to PyPI...")
            run_command("twine upload dist/*")
            print("✅ Successfully uploaded MOBPY 2.0.0 to PyPI!")
        else:
            print("⏸️  Upload cancelled.")
    
    # Post-release checklist
    print("\n📝 Post-release checklist:")
    print("   [ ] Create GitHub release tag: git tag v2.0.0")
    print("   [ ] Push tag: git push origin v2.0.0")
    print("   [ ] Update GitHub release notes")
    print("   [ ] Announce on social media/forums")
    print("   [ ] Update documentation if needed")
    
    print("\n✨ Release process complete!")


if __name__ == "__main__":
    main()