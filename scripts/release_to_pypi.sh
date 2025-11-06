#!/bin/bash

# MOBPY Release Script for PyPI
# Dynamically reads version from pyproject.toml

set -e  # Exit on error

# Extract version from pyproject.toml
VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' pyproject.toml || echo "unknown")

echo "🚀 MOBPY Release Script v${VERSION}"
echo "================================"

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Must run from project root directory"
    exit 1
fi

# Check Python version
echo "📌 Checking Python version..."
python --version

# Verify version consistency
echo ""
echo "🔍 Verifying version consistency..."
echo "   pyproject.toml: ${VERSION}"

# Check __init__.py version
if [ -f "src/MOBPY/__init__.py" ]; then
    INIT_VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' src/MOBPY/__init__.py || echo "unknown")
    echo "   __init__.py:    ${INIT_VERSION}"
    
    if [ "$VERSION" != "$INIT_VERSION" ]; then
        echo "❌ Version mismatch! Fix __init__.py to match ${VERSION}"
        exit 1
    fi
fi

echo "   ✅ Version consistency check passed"

# Clean previous builds
echo ""
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
echo "   Version: ${VERSION}"
echo ""
read -p "Upload to TestPyPI first? (recommended) [y/N]: " test_upload

if [[ $test_upload =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo "   Test install with: pip install --index-url https://test.pypi.org/simple/ MOBPY==${VERSION}"
    echo ""
    read -p "Continue to production PyPI? [y/N]: " prod_upload
    
    if [[ $prod_upload =~ ^[Yy]$ ]]; then
        echo "📤 Uploading to PyPI..."
        twine upload dist/*
        echo "✅ Successfully uploaded MOBPY ${VERSION} to PyPI!"
    else
        echo "⏸️  Production upload cancelled."
    fi
else
    read -p "Upload directly to PyPI? [y/N]: " direct_upload
    
    if [[ $direct_upload =~ ^[Yy]$ ]]; then
        echo "📤 Uploading to PyPI..."
        twine upload dist/*
        echo "✅ Successfully uploaded MOBPY ${VERSION} to PyPI!"
    else
        echo "⏸️  Upload cancelled."
    fi
fi

echo ""
echo "📝 Post-release checklist:"
echo "   [ ] Create GitHub release tag: git tag v${VERSION}"
echo "   [ ] Push tag: git push origin v${VERSION}"
echo "   [ ] Update GitHub release notes"
echo "   [ ] Announce on social media/forums"
echo "   [ ] Update documentation if needed"

echo ""
echo "✨ Release process complete!"