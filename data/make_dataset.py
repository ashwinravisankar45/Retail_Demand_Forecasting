import numpy as np
import pandas as pd

def main(seed: int = 7, n_stores: int = 12, n_skus: int = 35, weeks: int = 104, out_path: str = "data/weekly_demand.csv"):
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2023-01-02")  # Monday
    week_starts = pd.date_range(start, periods=weeks, freq="W-MON")

    stores = [f"S{100+i}" for i in range(n_stores)]
    skus = [f"SKU{1000+i}" for i in range(n_skus)]

    sku_base = rng.lognormal(mean=2.3, sigma=0.45, size=n_skus) * 10
    sku_price = np.clip(rng.normal(12, 4, size=n_skus), 3, 30)
    sku_elasticity = np.clip(rng.normal(1.2, 0.3, size=n_skus), 0.5, 2.0)

    rows = []
    for store in stores:
        store_factor = rng.normal(1.0, 0.12)
        for j, sku in enumerate(skus):
            base = sku_base[j] * store_factor
            price0 = sku_price[j]
            elast = sku_elasticity[j]
            for dt in week_starts:
                weekofyear = dt.isocalendar().week
                season = 1.0 + 0.18*np.sin(2*np.pi*(weekofyear/52.0)) + 0.08*np.cos(2*np.pi*(weekofyear/26.0))
                holiday = 1 if (dt.month==11 and dt.day>=15) or (dt.month==12) else 0
                promo = rng.binomial(1, p=0.18 + 0.08*holiday)
                price = np.clip(price0 * rng.normal(1.0, 0.04) * (0.92 if promo else 1.0), 1.5, 60)
                demand_mean = base * season * (1.18 if promo else 1.0) * ((price0/price) ** elast) * (1.12 if holiday else 1.0)
                units = max(0, int(np.round(demand_mean * (1 + rng.normal(0, 0.18)))))
                rows.append([dt.date().isoformat(), store, sku, float(price), int(promo), int(holiday), units])

    df = pd.DataFrame(rows, columns=["week_start", "store_id", "sku_id", "price", "promo", "holiday", "units"])
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    main()
