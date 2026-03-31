# Movie Recommendation System — BIL573 Project

A comprehensive movie recommendation system built on the **MovieLens 1M** dataset. The project explores and compares multiple collaborative filtering approaches — from classical baselines to deep learning models — and proposes a hybrid method that combines content and collaborative signals.

## Project Structure

```
project/
├── notebooks/
│   └── my_notebook.ipynb        # Main notebook (all experiments)
├── dataset/                     # MovieLens 1M raw data (.dat files)
│   ├── ratings.dat
│   ├── movies.dat
│   ├── users.dat
├── requirements.txt
└── README.md
```

| Path | Description |
|------|-------------|
| `notebooks/my_notebook.ipynb` | Single, self-contained notebook covering data loading, preprocessing, model training, tuning, and evaluation. |
| `dataset/` | MovieLens 1M dataset — ~1M ratings from 6,040 users on ~3,900 movies, plus actor/star metadata. |

## Notebook Overview (`my_notebook.ipynb`)

The notebook is organized into the following sections:

1. **Imports and Configuration** — library imports, device setup (CPU/GPU), and constants.
2. **Utility Functions** — helper functions for metrics (RMSE, MAE, Precision@K) and evaluation.
3. **Preprocessing Dataset** — loading `.dat` files, merging metadata, encoding features, and train/test split.
4. **Baseline Models**
   - **BaselineOnly** — global mean + user/item bias (Surprise).
   - **Memory-Based CF** — User kNN and Item kNN.
   - **Model-Based CF** — Pure SVD, SVD++, Neural Collaborative Filtering (NCF), and LightGCN.
5. **Proposed Method** — a hybrid approach combining collaborative filtering and content-based signals.
6. **Final Model Comparison** — side-by-side evaluation of all models with visualizations.

## Dataset

The project uses the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset:

- **1,000,209** ratings (1–5 scale) from **6,040** users on **3,952** movies.
- Additional metadata: user demographics (age, gender, occupation), movie genres, actor/star information.

> **Note:** Dataset files are excluded from version control via `.gitignore`. Download the dataset and place the `.dat` files in the `dataset/` directory before running the notebook.

The link to the dataset is: https://grouplens.org/datasets/movielens/1m/

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation and analysis |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, TF-IDF, Ridge regression, train/test split, metrics |
| `scikit-surprise` | SVD, SVD++, BaselineOnly, KNN-based collaborative filtering |
| `optuna` | Bayesian hyper-parameter optimization |
| `torch` | Neural models (NCF, LightGCN, proposed hybrid) |
| `scipy` | Sparse matrix operations |

## Getting Started

```bash
# 1. Clone the repository
git clone <repo-url> && cd project

# 2. Create and activate a virtual environment
python -m venv .env && source .env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place MovieLens 1M dataset files in dataset/

# 5. Open the notebook
jupyter notebook notebooks/my_notebook.ipynb
```

