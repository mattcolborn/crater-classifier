.PHONY: install activate train evaluate clean help lint format typecheck sort check-all

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with Poetry"
	@echo "  make train      - Run the training script"
	@echo "  make evaluate   - Run the evaluation script"
	@echo "  make clean      - Remove cached Python files"
	@echo "  make lint       - Check code with flake8"
	@echo "  make format     - Auto-format code with black"
	@echo "  make typecheck  - Check types with mypy"
	@echo "  make sort       - Sort imports with isort"
	@echo "  make check-all  - Run all checks"

install:
	poetry install || true

train:
	poetry run python main.py

evaluate:
	poetry run python -c "from src.crater_classifier.evaluate import evaluate; evaluate()"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	poetry run flake8 src/

format:
	poetry run black src/

typecheck:
	poetry run mypy src/

sort:
	poetry run isort src/

check-all:
	poetry run isort src/
	poetry run black src/
	poetry run flake8 src/
	poetry run mypy src/
