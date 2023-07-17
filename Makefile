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

.PHONY: remove-all
remove-all:
	rm -rf temp
	rm -rf mlruns
	rm -rf models

.PHONY: split-data
split-data:
	@$(USE_VENV)
	python src/data/split_data.py \
	--input-dir data --output-dir temp/splitted \
	--data-file "heart.csv" --label HeartDisease

.PHONY: preprocess-data
preprocess-data:
	@$(USE_VENV)
	python src/features/preprocess_heart.py \
	--input-dir temp/splitted  --output-dir temp/preprocessed \
	--data-file train_heart.csv --models-dir models
	python src/features/preprocess_heart.py \
	--input-dir temp/splitted  --output-dir temp/preprocessed \
	--data-file test_heart.csv --models-dir models --scaler-file heart_scaler.pkl

.PHONY: train
train:
	@$(USE_VENV)
	python src/models/train_heart.py \
	--train-file temp/preprocessed/train_heart.csv \
	--test-file temp/preprocessed/test_heart.csv \
	--label HeartDisease --scaler-file models/heart_scaler.pkl