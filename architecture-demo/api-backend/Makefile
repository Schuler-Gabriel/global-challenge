# API Backend Commands
.PHONY: setup run test

setup:
	python -m pip install -r requirements.txt

run:
	uvicorn main:app --reload

test:
	pytest tests/ -v
