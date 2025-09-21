#!/bin/bash
# Fix Formatting Script
# Automatically fixes formatting issues

set -e

echo "🔧 Fixing Code Formatting Issues..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Install dependencies if needed
if ! command -v black &> /dev/null; then
    echo "Installing code quality tools..."
    pip install black isort flake8
fi

# Fix formatting with Black
echo -e "${YELLOW}🎨 Running Black formatter...${NC}"
black src/ tests/ --line-length=120

# Fix import sorting with isort
echo -e "${YELLOW}📦 Running isort...${NC}"
isort src/ tests/ --profile=black --line-length=120

# Fix trailing whitespace and other issues
echo -e "${YELLOW}🧹 Cleaning up whitespace...${NC}"
find src/ tests/ -name "*.py" -exec sed -i 's/[[:space:]]*$//' {} \;

# Add newlines at end of files
echo -e "${YELLOW}📄 Adding newlines at end of files...${NC}"
find src/ tests/ -name "*.py" -exec sh -c 'if [ -s "$1" ] && [ "$(tail -c1 "$1")" != "" ]; then echo "" >> "$1"; fi' _ {} \;

echo -e "${GREEN}✅ Formatting fixes applied!${NC}"
echo -e "${YELLOW}💡 Run 'scripts/code-quality.sh' to verify all issues are fixed${NC}"
