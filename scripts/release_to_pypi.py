#!/usr/bin/env python3
"""
MOBPY Release Script for PyPI
Version 2.1.0

This script automates the release process for MOBPY to PyPI.
"""

import os
import sys
import shutil
import subprocess
import re
from pathlib import Path


def get_package_version():
    """Extract version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found!")
        sys.exit(1)
    
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    
    print("❌ Could not find version in pyproject.toml")
    sys.exit(1)


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
    # Get version dynamically
    version = get_package_version()
    
    print(f"🚀 MOBPY Release Script v{version}")
    print("=" * 40)
    
    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Must run from project root directory")
        sys.exit(1)
    
    # Check Python version
    print("\n📌 Checking Python version...")
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    # Verify version consistency
    print(f"\n🔍 Verifying version consistency...")
    print(f"   pyproject.toml: {version}")
    
    # Check __init__.py version
    init_file = Path("src/MOBPY/__init__.py")
    if init_file.exists():
        init_content = init_file.read_text()
        init_match = re.search(r'__version__\s*=\s*"([^"]+)"', init_content)
        if init_match:
            init_version = init_match.group(1)
            print(f"   __init__.py:    {init_version}")
            if init_version != version:
                print(f"❌ Version mismatch! Fix __init__.py to match {version}")
                sys.exit(1)
        else:
            print("⚠️  Could not verify __init__.py version")
    
    print("   ✅ Version consistency check passed")
    
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
    print(f"   Version: {version}")
    print()
    
    test_upload = input("Upload to TestPyPI first? (recommended) [y/N]: ").strip().lower()
    
    if test_upload == 'y':
        print("\n📤 Uploading to TestPyPI...")
        run_command("twine upload --repository testpypi dist/*")
        
        print("\n✅ Uploaded to TestPyPI!")
        print("   Test install with:")
        print(f"   pip install --index-url https://test.pypi.org/simple/ MOBPY=={version}")
        print()
        
        prod_upload = input("Continue to production PyPI? [y/N]: ").strip().lower()
        
        if prod_upload == 'y':
            print("\n📤 Uploading to PyPI...")
            run_command("twine upload dist/*")
            print(f"✅ Successfully uploaded MOBPY {version} to PyPI!")
        else:
            print("⏸️  Production upload cancelled.")
    else:
        direct_upload = input("Upload directly to PyPI? [y/N]: ").strip().lower()
        
        if direct_upload == 'y':
            print("\n📤 Uploading to PyPI...")
            run_command("twine upload dist/*")
            print(f"✅ Successfully uploaded MOBPY {version} to PyPI!")
        else:
            print("⏸️  Upload cancelled.")
    
    # Post-release checklist
    print("\n📝 Post-release checklist:")
    print(f"   [ ] Create GitHub release tag: git tag v{version}")
    print(f"   [ ] Push tag: git push origin v{version}")
    print("   [ ] Update GitHub release notes")
    print("   [ ] Announce on social media/forums")
    print("   [ ] Update documentation if needed")
    
    print("\n✨ Release process complete!")


if __name__ == "__main__":
    main()