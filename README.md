# Global innovation index (gii) predictor

## Project overview
This research identifies driving factors of innovation using machine learning. It utilises the 2024 WIPO GII data and World Bank edStats to analyse how investments and outputs contribute to national innovation levels.


## Methodology

### Data cleaning and imputation
Dropped columns with >30% missing data and used knn to fill remaining gaps. Focused on indicator-level and demographic data.

### Data augmentation
Generated 800 synthetic points via 0.1 noise ratio to address limited national-level data. Similarity was verified via statistical tests while keeping the target variable clean to protect the coefficients.

### Feature selection and modeling
Used elasticnetcv with recursive feature elimination to find the top 15 variables. Trained a lasso model with inverse octile weighting and grid search for alpha optimisation.

### Validation
Data was stratified into 43 training and 40 validation rows, with 50 real-world rows held for final model verification.


## Directory structure and relationships
Execute:

```text
.
├── data/
│   ├── raw/                # original source files
│   └── processed/          # imputed and synthetic sets
├── notebooks/
│   ├── 01_exploration/     # cleaning and imputation
│   ├── 02_augmentation/    # noise and similarity tests
│   ├── 03_selection/       # feature elimination
│   └── 04_modeling/        # lasso training and tuning
├── figure_exports/         # plots and performance metrics
├── requirements.txt        # python dependencies
└── README.md               # project documentation
```

## Reproduction steps

1. Setup: Install requirements.txt.
2. Preprocess: Clean data and apply KNN.
3. Augment: Generate 800 synthetic points and verify similarity.
4. Select: Run rfe via elasticnetcv.
5. Train: Run lasso with weighting and grid search.
6. Validate: Test against stratified and holdout sets.
