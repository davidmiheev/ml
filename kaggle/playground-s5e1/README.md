# Playground Series S5E1: Forecasting Sticker Sales

[Competition](https://www.kaggle.com/competitions/playground-series-s5e1) ·
January 2025 · metric: MAPE

Daily sticker sales for 6 countries × 3 stores × 5 products (90 series), with
2010–2016 as training data. The task is to forecast 2017–2019. About 3.9% of the
training target (8,871 rows) is missing, and two series are missing entirely.

## My notebooks (`my-notebooks/`)

These are working notebooks kept as they were, with dead ends and
commented-out experiments left in.

| Notebook | What it tries |
| --- | --- |
| [`01-groupby-baselines-and-gbdt-ensemble`](my-notebooks/01-groupby-baselines-and-gbdt-ensemble.ipynb) | Group-mean baselines keyed on `country-store-product-season-dayofweek`, a reusable sklearn preprocessing/pipeline builder, and an Optuna-tuned XGBoost + LightGBM + CatBoost ensemble with K-fold MAPE CV. |
| [`02-knn-imputation-catboost-mape`](my-notebooks/02-knn-imputation-catboost-mape.ipynb) | Calendar and cyclic (sin/cos day-of-year) features. Fills missing targets with a distance-weighted KNN regressor, then fits CatBoost with a MAPE loss on the completed data. |
| [`03-knn-baseline-catboost-residual`](my-notebooks/03-knn-baseline-catboost-residual.ipynb) | Two-stage model: KNN on one half of the data supplies a `baseline` to CatBoost on the other half, so the boosted trees learn the residual. |

## Community notebooks studied (`community/`)

These are copies of public notebooks, kept to read and run. All credit goes to
their authors.

| Notebook | Author | Why it's here |
| --- | --- | --- |
| [EDA and Linear Regression Baseline](https://www.kaggle.com/code/cabaxiom/s5e1-eda-and-linear-regression-baseline) | Cabaxiom | Shows that the 90 series collapse into one aggregate. Country share follows **GDP per capita**, store share is constant, and product share is a sinusoid. Forecast the total, then disaggregate. |
| [Previous Years Baseline: No Model](https://www.kaggle.com/code/cabaxiom/s5e1-previous-years-baseline-no-model) | Cabaxiom | The same decomposition with no learned model at all: a day-of-year mean with a day-of-week correction. A strong baseline that's hard to beat. |
| [Transformer Starter (LB 0.052)](https://www.kaggle.com/code/cdeotte/transformer-starter-lb-0-052) | Chris Deotte | Normalises by GDP and store ratio, then uses a WaveNet + Transformer sequence model on random crops, with a NaN-indicator channel. Shows how to use a deep time-series model on a tabular-looking problem. |

**Main takeaway:** the winning direction was structure, not model capacity.
Find the multiplicative factors (GDP, store, product seasonality), forecast one
clean aggregate, and the fancy models become optional. My GBDT notebooks
worked on the raw 90 series and fought the noise those factors explain.

## Data

```bash
cd kaggle/playground-s5e1
kaggle competitions download -c playground-series-s5e1 && unzip playground-series-s5e1.zip
```

My notebooks read `train.csv` / `test.csv` from the working directory or from
`/kaggle/input/playground-series-s5e1/`. The community notebooks also need GDP per capita. The Cabaxiom notebooks read
the Kaggle dataset `world-gdpgdp-gdp-per-capita-and-annual-growths`, and the
Transformer Starter queries the World Bank API.
