import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out["weekofyear"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_start"].dt.month
    out["year"] = out["week_start"].dt.year
    out["sin_woy"] = np.sin(2*np.pi*out["weekofyear"]/52.0)
    out["cos_woy"] = np.cos(2*np.pi*out["weekofyear"]/52.0)
    return out

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "weekly_demand.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing dataset. Run: python data/make_dataset.py")

    df = pd.read_csv(data_path)
    df = add_time_features(df)

    cutoff = pd.to_datetime(df["week_start"]).quantile(0.8)
    train = df[pd.to_datetime(df["week_start"]) <= cutoff].copy()
    test = df[pd.to_datetime(df["week_start"]) > cutoff].copy()

    X_train = train.drop(columns=["units"])
    y_train = train["units"]
    X_test = test.drop(columns=["units"])
    y_test = test["units"]

    cat = ["store_id","sku_id"]
    num = ["price","promo","holiday","weekofyear","month","year","sin_woy","cos_woy"]

    pre = ColumnTransformer([
       ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", "passthrough", num)
    ])

    model = Pipeline([
        ("prep", pre),
        ("model", HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, max_iter=300, random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))

    (root / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "mae": mae}, root / "models" / "demand_forecaster.joblib")

    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "metrics.json").write_text(pd.Series({"mae": mae}).to_json(), encoding="utf-8")

    # Example plot (one store×sku)
    sample = test[(test["store_id"]=="S100") & (test["sku_id"]=="SKU1000")].copy()
    if len(sample) > 0:
        sample = add_time_features(sample)
        plt.figure(figsize=(9,4))
        plt.plot(pd.to_datetime(sample["week_start"]), sample["units"], label="Actual")
        plt.plot(pd.to_datetime(sample["week_start"]), model.predict(sample.drop(columns=["units"])), label="Forecast")
        plt.title("Example Forecast — S100 × SKU1000")
        plt.xlabel("Week"); plt.ylabel("Units")
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / "reports" / "example_forecast.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("Saved models/demand_forecaster.joblib")
    print("MAE:", round(mae, 2))

if __name__ == "__main__":
    main()
