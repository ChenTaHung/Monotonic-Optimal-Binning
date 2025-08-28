#!/bin/bash

# MOBPY Release Script for PyPI
# Version 2.0.0

set -e  # Exit on error

echo "🚀 MOBPY Release Script v2.0.0"
echo "================================"

# Check Python version
echo "📌 Checking Python version..."
python --version

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ src/*.egg-info

# Upgrade build tools
echo "⬆️ Upgrading build tools..."
pip install --upgrade pip setuptools wheel twine build

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --tb=short || {
    echo "❌ Tests failed. Please fix before releasing."
    exit 1
}

# Build the package
echo "📦 Building package..."
python -m build

# Check the package
echo "🔍 Checking package with twine..."
twine check dist/*

# Display package contents
echo "📋 Package contents:"
ls -la dist/

# Ask for confirmation
echo ""
echo "⚠️  Ready to upload to PyPI?"
echo "   Package: MOBPY"
echo "   Version: 2.0.0"
echo ""
read -p "Upload to TestPyPI first? (recommended) [y/N]: " test_upload

if [[ $test_upload =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo "   Test install with: pip install --index-url https://test.pypi.org/simple/ MOBPY"
    echo ""
    read -p "Continue to production PyPI? [y/N]: " prod_upload
    
    if [[ $prod_upload =~ ^[Yy]$ ]]; then
        echo "📤 Uploading to PyPI..."
        twine upload dist/*
        echo "✅ Successfully uploaded MOBPY 2.0.0 to PyPI!"
    else
        echo "⏸️  Production upload cancelled."
    fi
else
    read -p "Upload directly to PyPI? [y/N]: " direct_upload
    
    if [[ $direct_upload =~ ^[Yy]$ ]]; then
        echo "📤 Uploading to PyPI..."
        twine upload dist/*
        echo "✅ Successfully uploaded MOBPY 2.0.0 to PyPI!"
    else
        echo "⏸️  Upload cancelled."
    fi
fi

echo ""
echo "📝 Post-release checklist:"
echo "   [ ] Create GitHub release tag: git tag v2.0.0"
echo "   [ ] Push tag: git push origin v2.0.0"
echo "   [ ] Update GitHub release notes"
echo "   [ ] Announce on social media/forums"
echo "   [ ] Update documentation if needed"