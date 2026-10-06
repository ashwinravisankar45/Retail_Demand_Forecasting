Retail Demand Forecasting

A machine learning pipeline for forecasting weekly retail product demand using historical sales, pricing, promotion, holiday, store, SKU, and time-based features.

The project demonstrates an end-to-end machine learning workflow covering data preparation, feature engineering, preprocessing, model training, chronological evaluation, model serialization, and forecast visualization.

Overview

Accurate demand forecasting helps retailers make better decisions around:

Inventory planning

Product availability

Pricing

Promotions

Supply planning

This project builds a supervised machine learning pipeline that predicts weekly product demand (units) from historical retail data.

Project Workflow

Raw Retail Data
      │
      ▼
Data Preparation
      │
      ▼
Time-Based Feature Engineering
      │
      ▼
Categorical & Numerical Preprocessing
      │
      ▼
HistGradientBoostingRegressor
      │
      ▼
Chronological Evaluation
      │
      ▼
Forecast Visualization
      │
      ▼
Exported ML Model

Key Highlights

Built a complete retail demand forecasting pipeline using Python

Engineered calendar and seasonal features from weekly dates

Encoded store and SKU identifiers using OneHotEncoder

Trained a HistGradientBoostingRegressor

Used a chronological 80/20 train-test split to preserve temporal ordering

Evaluated predictions using Mean Absolute Error (MAE)

Achieved a test MAE of 12.58

Serialized the trained forecasting pipeline using joblib

Generated actual-vs-predicted demand visualizations

Organized the project into reproducible data, model, notebook, report, and source-code directories

Dataset

The project uses weekly retail demand data containing the following fields:

Feature

Description

week_start

Start date of the sales week

store_id

Store identifier

sku_id

Product/SKU identifier

price

Product price

promo

Promotion indicator

holiday

Holiday indicator

units

Target variable representing weekly demand

Target Variable

units

The model learns the relationship between the available retail and time-based features and the number of units demanded.

Feature Engineering

The week_start date is transformed into multiple temporal features:

Week of year

Month

Year

Sine transformation of week number

Cosine transformation of week number

Seasonal Encoding

The sine and cosine transformations provide a cyclical representation of the week-of-year feature, allowing the model to capture recurring annual patterns more effectively.

weekofyear
      │
      ├── sin_woy
      │
      └── cos_woy

Machine Learning Pipeline

Categorical Features

store_id
sku_id

These categorical variables are transformed using:

OneHotEncoder

Numerical Features

price
promo
holiday
weekofyear
month
year
sin_woy
cos_woy

Model

The forecasting model is:

HistGradientBoostingRegressor

Configuration:

max_depth      = 8
learning_rate  = 0.08
max_iter       = 300
random_state   = 42

Train-Test Strategy

Because this is a demand forecasting problem, the data is split chronologically rather than randomly.

Historical Timeline
│
├────────────── 80% ──────────────┤── 20% ──┤
│             Training            │ Testing │
└─────────────────────────────────┴─────────┘

This prevents future observations from being randomly mixed into the training data and provides a more appropriate evaluation setup for time-dependent demand forecasting.

Model Performance

The model is evaluated using:

Mean Absolute Error — MAE

MAE = 12.58

MAE measures the average absolute difference between actual and predicted demand.

Lower MAE indicates better predictive accuracy.

The evaluation metric is stored in:

reports/metrics.json

Forecast Visualization

The project generates an actual-vs-predicted demand visualization for:

Store: S100
SKU: SKU1000

The visualization compares:

Actual weekly demand

Predicted weekly demand

Output:

reports/example_forecast.png

Model Export

The trained forecasting pipeline is serialized using joblib:

models/demand_forecaster.joblib

The exported object contains:

{
    "model": model,
    "mae": mae
}

This allows the trained model and its evaluation result to be retained for future inference without retraining the model.

Project Structure

Retail_Demand_Forecasting/
│
├── data/
│   ├── make_dataset.py
│   └── weekly_demand.csv
│
├── models/
│   └── demand_forecaster.joblib
│
├── notebooks/
│   └── demand_forecasting.ipynb
│
├── reports/
│   ├── example_forecast.png
│   └── metrics.json
│
├── src/
│   ├── forecast.py
│   └── train_and_export.py
│
├── .gitignore
├── README.md
└── requirements.txt

Technologies & Tools

Category

Technologies

Programming

Python

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn

Visualization

Matplotlib, Seaborn

Model Serialization

Joblib

Development

Jupyter Notebook, VS Code

Version Control

Git, GitHub

Installation

1. Clone the repository

git clone https://github.com/ashwinravisankar45/Retail_Demand_Forecasting.git
cd Retail_Demand_Forecasting

2. Create a virtual environment

Windows PowerShell

python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

Running the Project

Ensure the dataset is available at:

data/weekly_demand.csv

Run the training pipeline:

python src/train_and_export.py

Successful execution generates:

models/demand_forecaster.joblib
reports/metrics.json
reports/example_forecast.png

The terminal displays the resulting model performance:

Saved models/demand_forecaster.joblib
MAE: 12.58

Key Concepts Demonstrated

This project demonstrates practical knowledge of:

Data preprocessing

Feature engineering

Time-series feature extraction

Cyclical feature encoding

Categorical variable encoding

Gradient boosting regression

Chronological train-test splitting

Model evaluation

Model serialization

Demand forecasting

Data visualization

Reproducible ML project organization

Future Improvements

Potential extensions include:

Adding lag-based demand features

Adding rolling demand statistics

Comparing XGBoost and LightGBM

Hyperparameter optimization

Comparing multiple forecasting algorithms

Incorporating additional historical demand information

Building an interactive forecasting dashboard

Exposing the model through an API

Automating future-demand predictions

Author

Ashwin

B.Tech Computer Science & Engineering
Data Science | Artificial Intelligence | Machine Learning

⭐ If you find this project useful, consider giving the repository a star.