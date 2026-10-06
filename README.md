# Retail Demand Forecasting

A machine learning project for forecasting weekly retail product demand using historical sales-related features such as price, promotions, holidays, store, SKU, and time-based patterns.

## Overview

Retail demand forecasting helps businesses estimate future product demand and make better decisions around inventory, pricing, promotions, and supply planning.

This project builds a machine learning pipeline that:

- Processes weekly retail demand data
- Creates time-based features
- Encodes categorical store and SKU information
- Trains a `HistGradientBoostingRegressor`
- Evaluates the model using Mean Absolute Error (MAE)
- Saves the trained forecasting model
- Generates an example actual-vs-forecast visualization

## Project Structure

```text
retail_demand_forecasting/
│
├── data/
│   └── weekly_demand.csv
│
├── models/
│   └── demand_forecaster.joblib
│
├── notebooks/
│
├── reports/
│   ├── metrics.json
│   └── example_forecast.png
│
├── src/
│   ├── forecast.py
│   └── train_and_export.py
│
├── README.md
└── requirements.txt
```

## Dataset

The model uses weekly retail demand data containing features such as:

| Feature | Description |
|---|---|
| `week_start` | Start date of the sales week |
| `store_id` | Store identifier |
| `sku_id` | Product/SKU identifier |
| `price` | Product price |
| `promo` | Promotion indicator |
| `holiday` | Holiday indicator |
| `units` | Target variable representing demand |

## Feature Engineering

The project creates additional time-based features from `week_start`:

- Week of year
- Month
- Year
- Sine transformation of week number
- Cosine transformation of week number

The sine and cosine transformations help the model capture seasonal patterns across the year.

## Machine Learning Pipeline

The preprocessing pipeline contains:

### Categorical Features

- `store_id`
- `sku_id`

These features are transformed using `OneHotEncoder`.

### Numerical Features

- `price`
- `promo`
- `holiday`
- `weekofyear`
- `month`
- `year`
- `sin_woy`
- `cos_woy`

### Model

The forecasting model uses:

```text
HistGradientBoostingRegressor
```

Configuration:

```text
max_depth = 8
learning_rate = 0.08
max_iter = 300
random_state = 42
```

## Train-Test Split

The dataset is divided chronologically rather than randomly.

- 80% of the timeline → Training data
- 20% of the timeline → Testing data

This approach is suitable for demand forecasting because future observations should not be used to train the model.

## Evaluation

The model is evaluated using **Mean Absolute Error (MAE)**.

The current model produced:

```text
MAE: 12.58
```

A lower MAE indicates that the model's demand predictions are closer to the actual demand values.

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

## Running the Project

Make sure the dataset is available at:

```text
data/weekly_demand.csv
```

Then run:

```powershell
python src/train_and_export.py
```

Successful execution will generate:

```text
models/demand_forecaster.joblib
reports/metrics.json
reports/example_forecast.png
```

The terminal will display the model performance:

```text
Saved models/demand_forecaster.joblib
MAE: 12.58
```

## Forecast Visualization

The project generates an example forecast for:

```text
Store: S100
SKU: SKU1000
```

The visualization compares:

- Actual weekly demand
- Predicted weekly demand

The generated chart is saved as:

```text
reports/example_forecast.png
```

## Model Export

The trained pipeline is saved using `joblib`:

```text
models/demand_forecaster.joblib
```

The saved object contains:

```python
{
    "model": model,
    "mae": mae
}
```

This allows the trained model to be reused for future predictions without retraining.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Key Concepts

This project demonstrates practical applications of:

- Data preprocessing
- Feature engineering
- Time-series feature extraction
- Categorical encoding
- Gradient boosting
- Regression
- Model evaluation
- Model serialization
- Demand forecasting
- Data visualization

## Future Improvements

Potential improvements include:

- Adding lag-based demand features
- Adding rolling average features
- Testing XGBoost or LightGBM
- Hyperparameter tuning
- Comparing multiple forecasting models
- Adding additional historical demand features
- Building an interactive forecasting dashboard
- Deploying the model as an API
- Adding automated future-demand predictions

## Author

Ashwin
B.Tech Computer Science & Engineering  
Data Science | Artificial Intelligence | Machine Learning

---

⭐ If you find this project useful, consider giving the repository a star.