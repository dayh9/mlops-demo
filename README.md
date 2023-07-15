# Remember to run any scripts from top main dir!
---
# DVC
## Init DVC
Init DVC - creates .dvc

```bash
dvc init
git commit -m "Init DVC"
```

## Set up remote DVC data storage

```bash
dvc remote add -d remote_storage ./dvc_remote
dvc config core.autostage true
git add .dvc/config
git commit -m "Configure remote storage"
```

## Track data

```bash
dvc add data/heart.csv
git add data/heart.csv.dvc data/.gitignore
git commit -m "Dataset v1"
git tag "v1"
```
Running `dvc add` creates a `heart.csv.dvc` to track git and which DVC uses to detected changes in the data. The `.gitignore` is also updated to ignore the data itself from git tracking (Git tracks only the `heart.csv.dvc` file). The `.dvc` file contains the file hash and some file metadata.

## Push data

```bash
dvc push
```
Pushes data to remote storage.

## Data changes

To check:
```bash
dvc status
```

To update
```bash
dvc add data/heart.csv
```

To trach changes with git:
```bash
git add data/heart.csv.dvc
git commit -m "Dataset v2"
git tag "v2"
```

To push updated data
```bash
dvc push
```

## Switching between dataset versions
Switching between dataset versions involves a combination of `git checkout` and `dvc checkout` (or `dvc pull`). The correct version of the `heart.csv.dvc` file is loaded into workspace via `git checkout` and running `dvc checkout` then pulls the associated data from our local cache (to pull the data from the remote, you would run `dvc pull`).
```bash
git checkout tags/v1 
dvc checkout
```

## Order of python scripts to mimic pipeline flow
1. Split data and save to separate files
`python src/data/split_data.py --input-dir data --output-dir temp/splitted --file "heart.csv" --label HeartDisease`
2. Transform train data, save x_train, y_train and scaler
`python src/features/preprocess_heart.py --input-dir temp/splitted  --output-dir temp/preprocessed --file train_heart.csv --models-dir models`
3. Transform test data using saved scaler and save x_test, y_test
`python src/features/preprocess_heart.py --input-dir temp/splitted  --output-dir temp/preprocessed --file test_heart.csv --models-dir models --scaler-file heart_scaler.pkl`