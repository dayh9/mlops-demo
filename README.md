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