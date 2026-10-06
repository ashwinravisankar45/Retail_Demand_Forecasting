import pandas as pd
import numpy as np
import joblib

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out["weekofyear"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_start"].dt.month
    out["year"] = out["week_start"].dt.year
    out["sin_woy"] = np.sin(2*np.pi*out["weekofyear"]/52.0)
    out["cos_woy"] = np.cos(2*np.pi*out["weekofyear"]/52.0)
    return out

if __name__ == "__main__":
    bundle = joblib.load("models/demand_forecaster.joblib")
    model = bundle["model"]
    future = pd.DataFrame([{
        "week_start": "2025-01-06",
        "store_id": "S100",
        "sku_id": "SKU1000",
        "price": 11.99,
        "promo": 0,
        "holiday": 0
    }])
    future = add_time_features(future)
    yhat = model.predict(future.drop(columns=[]))[0]
    print({"forecast_units": float(yhat)})
