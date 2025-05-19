tutorial:
	@echo "Makefile"

deps:
	@echo "Installing dependencies..."
	@pip install --upgrade -r requirements.txt

format:
	@echo "Formatting code..."
	@black .
	@isort .
	@flake8 --max-line-length=120 --extend-ignore=E203 --exclude .venv,build,dist .
