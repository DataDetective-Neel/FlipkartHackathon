# GridLock Alchemist — Flipkart Hackathon

Brief repository summary and quick usage notes for the Flipkart Hackathon submission.

## Description
This project contains training and ensembling scripts used to produce the final `submission.csv` for the Flipkart Hackathon. The repository includes data samples, model checkpoints, and helper files used during experimentation.

## Key files
- `GridLock_Alchemist.ipynb` — exploratory notebook
- `train_improved.py` — primary training script
- `train_ensemble.py` — ensembling / blending script
- `submission.csv` — final generated submission
- `dataset/` — contains `train.csv`, `test.csv`, and `sample_submission.csv`

## Requirements
- Python 3.8+ recommended
- Common packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`

Install typical dependencies with pip:

    pip install pandas numpy scikit-learn xgboost lightgbm catboost

## Quick usage
1. Prepare data in `dataset/` (already present in this repo).
2. Train a model:

    python train_improved.py

3. Run ensembling (if desired):

    python train_ensemble.py

4. The produced `submission.csv` is the file to submit.

## Notes
- See `portal_candidates/` for previous candidate blends and blending experiments.
- Inspect `catboost_info/` for CatBoost training logs and metrics.

If you want the README expanded (installation, examples, detailed argument docs), tell me what sections to add.
