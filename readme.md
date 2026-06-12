# Movie Recommendation System — Netflix Prize Dataset

Personalized recommendation engine built on the Netflix Prize dataset, comparing
Item-Based Collaborative Filtering and SVD (matrix factorization).

## Notebooks

1. **eda.ipynb** — Data loading & cleaning, dataset overview, rating distribution,
   user activity, movie popularity/quality, sparsity analysis. Outputs `ratings.csv`
   and `movies.csv`.
2. **cf_svd_map_eval.ipynb** — Subsamples active users (≥100 ratings) and popular
   movies (≥40 ratings), performs a per-user 80/20 train/test split, builds an
   Item-Based CF model (cosine similarity) and an SVD model (Surprise), and
   evaluates both.

## Dataset Summary

- 24,053,764 ratings | 470,758 users | 4,499 movies
- Sparsity: 98.86%
- Avg rating: 3.60/5 (mode = 4)

## Methodology

- **Sample:** 5,000 active users, movies with ≥40 ratings → 1,813 movies, 901,682 ratings
- **Split:** per-user 80/20 (avoids zero-history test users)
- **Relevance:** rating ≥ 3.5
- **Recommendations:** Top-10 per user from shared candidate pool

## Results

| Model         | RMSE   | MAP@10 |
|---------------|--------|--------|
| Item-Based CF | 0.9596 | 0.2323 |
| SVD           | 0.8475 | 0.0213 |

**Item-CF** wins on ranking quality (MAP@10); **SVD** wins on rating-prediction
accuracy (RMSE).

## Setup

```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn
```

Run `eda.ipynb` first (produces `../data/ratings.csv` and `../data/movies.csv`),
then run `cf_svd_map_eval.ipynb`.