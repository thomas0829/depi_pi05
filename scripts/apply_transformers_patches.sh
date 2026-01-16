#!/bin/bash
# Apply transformers patches required for PI0.5 PyTorch implementation

set -e

echo "📦 Applying transformers patches for PI0.5..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Get site-packages path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

if [ ! -d "$SITE_PACKAGES/transformers" ]; then
    echo "❌ Error: transformers package not found at $SITE_PACKAGES/transformers"
    echo "Please install transformers==4.53.2 first:"
    echo "  pip install transformers==4.53.2"
    exit 1
fi

PATCHES_DIR="$PROJECT_ROOT/third_party/transformers_patches"
if [ ! -d "$PATCHES_DIR" ]; then
    echo "❌ Error: patches directory not found at $PATCHES_DIR"
    exit 1
fi

echo "   Source: $PATCHES_DIR"
echo "   Target: $SITE_PACKAGES/transformers"

# Copy patches
cp -rv "$PATCHES_DIR"/* "$SITE_PACKAGES/transformers/"

echo ""
echo "✅ Transformers patches applied successfully!"
echo ""
echo "⚠️  WARNING: These patches permanently modify your transformers installation."
echo "   To revert, reinstall transformers:"
echo "   pip install --force-reinstall transformers==4.53.2"
