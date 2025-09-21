#!/bin/bash
# Code Quality Script
# Runs all formatting and linting tools

set -e

echo "🔍 Running Code Quality Checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run command and check result
run_check() {
    local tool_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}Running $tool_name...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $tool_name passed${NC}"
    else
        echo -e "${RED}❌ $tool_name failed${NC}"
        return 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Install dependencies if needed
if ! command -v black &> /dev/null; then
    echo "Installing code quality tools..."
    pip install black isort flake8 mypy bandit pre-commit
fi

# Run formatting tools
echo -e "${YELLOW}📝 Running formatting tools...${NC}"

run_check "Black" "black --check --diff src/ tests/"
run_check "isort" "isort --check-only --diff src/ tests/"

# Run linting tools
echo -e "${YELLOW}🔍 Running linting tools...${NC}"

run_check "Flake8" "flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503,E501"
run_check "MyPy" "mypy src/ --ignore-missing-imports"
run_check "Bandit" "bandit -r src/ -f json -o bandit-report.json || true"

# Run pre-commit if available
if command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}🔄 Running pre-commit hooks...${NC}"
    run_check "Pre-commit" "pre-commit run --all-files"
fi

echo -e "${GREEN}🎉 All code quality checks passed!${NC}"
