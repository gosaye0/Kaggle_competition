# Ames Housing Price Prediction

> End-to-end machine learning pipeline for predicting house prices on the [Kaggle Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  


---

## Project Overview

This project implements a **full ML workflow** for tabular regression:

- **Data loading** (train/test, handling missing values)  
- **Feature engineering** (custom transformers & pipelines)  
- **Model training** (XGBoost inside a Scikit-Learn pipeline)  
- **Evaluation** (cross-validation with RMSE, MAE, R²)  
- **Inference** (predicting on Kaggle’s test set & exporting submission files)

The codebase is structured like a **production ML project** rather than a single notebook, making it easier to:

- Reproduce experiments  
- Train/evaluate consistently  
- Reuse preprocessing & inference logic  

---

##  Repository Structure

Ames Housing Project
Project Structure
project_root/
├── ames_housing/           # Core Python package
│   ├── __init__.py
│   ├── preprocessing.py    # Custom transformers & feature engineering
│   ├── data_loader.py      # Load raw/processed data
│   ├── model_training.py   # Training pipeline definition
│   ├── model_inference.py  # Load & use trained model
│   ├── model_evaluation.py # Cross-validation evaluation
│   ├── utils.py           # Utility helpers
│   └── constants.py       # Shared constants
│
├── scripts/               # CLI scripts
│   ├── train.py           # Train model & save pipeline
│   ├── evaluate.py        # Run CV & log metrics
│   └── predict.py         # Generate predictions on test set
│
├── data/                  # Local dataset storage (gitignored)
│   ├── raw/               # Downloaded Kaggle train/test.csv
│   └── processed/         # Any intermediate artifacts
│
├── outputs/               # Generated artifacts (gitignored)
│   ├── models/            # Saved trained models
│   ├── reports/           # CV metrics, logs
│   └── predictions/       # Submission files
│
├── notebooks/             # Jupyter notebooks (EDA, prototyping)
│   └── 01_eda_prototyping.ipynb
│
├── tests/                 # Unit & integration tests
│   ├── test_data_loader.py
│   └── test_preprocessing.py
│
├── requirements.txt       # Python dependencies
├── setup.py               # Installable package definition
└── README.md              # Project documentation

## Installation
Clone the repository:
`git clone https://github.com/<your-username>/ames-housing.git`
`cd ames-housing`

Install dependencies (editable mode):
`pip install -e .`

Download Kaggle dataset and place train.csv and test.csv into:
data/raw/

## Training
Run training from project root:
`python -m scripts.train`

This will:

Train the final pipeline on all training data
Save the trained model under outputs/models/

## Evaluation
Run cross-validation evaluation:
`python -m scripts.evaluate`

This will:

Perform 5-fold CV with metrics (RMSE, MAE, R²)
Save fold-level results to outputs/reports/cv_fold_details.csv
Save aggregate metrics to outputs/reports/cv_results.csv

output:

|       | RMSE                | MAE                | R²                  |
|-------|---------------------|--------------------|---------------------|
| Train | 14,532 ± 1,204      | 10,321 ± 842       | 0.95 ± 0.0102       |
| Test  | 22,876 ± 1,789      | 15,602 ± 1,201     | 0.89 ± 0.0134       |


## Inference / Prediction
Generate predictions for the Kaggle test set:
`python -m scripts.predict`

This will save a submission-ready file:
outputs/predictions/submission.csv

## Testing
Run tests with:
`pytest tests/`

## Author
**Gosaye Emshaw:** Machine Learning Engineer
