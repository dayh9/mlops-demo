# TODO:
# dvc commands to check status, update ds etc.
# run stop mlflow
# clear temp data storages

USE_VENV := . venv/bin/activate

.PHONY: create-venv
create-venv:
	python3.10 -m venv venv
	pip install -r requirements.txt

.PHONY: sort-imports
sort-imports: 
	@$(USE_VENV)
	isort --profile black src

.PHONY: format-python
format-python: 
	@$(USE_VENV)
	black src
