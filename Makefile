# Tabula Rasa - AI Training System
# Makefile for development and maintenance tasks

.PHONY: help install install-dev test lint format check clean build docs

# Default target
help:
	@echo "Tabula Rasa - AI Training System"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  check        Run all quality checks (lint + test)"
	@echo "  clean        Clean up temporary files"
	@echo "  build        Build package"
	@echo "  docs         Generate documentation"
	@echo "  pre-commit   Install pre-commit hooks"
	@echo "  pre-commit-run Run pre-commit on all files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/ -k "performance" -v

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/
	pydocstyle src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Quality Checks
check: format-check lint test

# Pre-commit
pre-commit:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/ dist/

# Build
build: clean
	python -m build

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation would be generated here"
	@echo "Consider using Sphinx or MkDocs for full documentation"

# Development workflow
dev-setup: install-dev pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything is working"

# CI/CD helpers
ci-test: test-unit test-integration
ci-lint: format-check lint
ci-check: ci-lint ci-test

# Quick development commands
quick-test:
	pytest tests/ -x -v --tb=short

quick-lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# System validation
validate-system:
	python -c "from src.training import ContinuousLearningLoop, MasterARCTrainer; print('✅ System validation passed')"

# Performance monitoring
monitor-performance:
	python -c "from src.monitoring import PerformanceMonitor; print('✅ Performance monitoring available')"

# Cache management
clear-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cache cleared"

# Database operations (if applicable)
db-migrate:
	@echo "Database migration would run here"
	@echo "Add your database migration commands"

db-reset:
	@echo "Database reset would run here"
	@echo "Add your database reset commands"

# Docker operations (if applicable)
docker-build:
	@echo "Docker build would run here"
	@echo "Add your Docker build commands"

docker-run:
	@echo "Docker run would run here"
	@echo "Add your Docker run commands"

# Release helpers
version-bump:
	@echo "Version bump would run here"
	@echo "Add your version bumping logic"

release-check: check build
	@echo "Release check complete"

# Help for specific targets
help-test:
	@echo "Test targets:"
	@echo "  test              - Run all tests with coverage"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-performance  - Run performance tests only"
	@echo "  quick-test        - Run tests with short output"

help-lint:
	@echo "Linting targets:"
	@echo "  lint         - Run all linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  format-check - Check if code is properly formatted"
	@echo "  quick-lint   - Run quick linting checks"

help-dev:
	@echo "Development targets:"
	@echo "  dev-setup    - Set up development environment"
	@echo "  check        - Run all quality checks"
	@echo "  clean        - Clean up temporary files"
	@echo "  validate-system - Validate system is working"
