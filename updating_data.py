import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_CSV = "nasa_data.csv"     
OUTPUT_CSV = "nasa_weather_extended_oct2025.csv"
END_DATE = "2025-10-31"
N_DAYS_MAX = None                       
N_SAMPLES = 8                           

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.upper()
df = df.sort_values(["CITY", "YEAR", "MO", "DY"]).reset_index(drop=True)

df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]))
numeric_cols = [
    "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M",
    "ALLSKY_SFC_UV_INDEX", "CLOUD_OD", "PRECTOTCORR",
    "PS", "WD2M", "ALLSKY_KT", "LATITUDE", "LONGITUDE"
]

cities = df["CITY"].unique()
df["MONTH"] = df["DATE"].dt.month
climatology = {} 
for city in tqdm(cities, desc="Computing climatology"):
    city_df = df[df["CITY"] == city]
    if city_df.empty:
        continue
    climatology[city] = {}
    grouped = city_df.groupby(city_df["MONTH"])
    for m in range(1, 13):
        if m in grouped.groups:
            g = grouped.get_group(m)
            m_mean = g[numeric_cols].mean(numeric_only=True)
            m_std = g[numeric_cols].std(numeric_only=True).fillna(0.0)
        else:
            m_mean = city_df[numeric_cols].mean(numeric_only=True)
            m_std = city_df[numeric_cols].std(numeric_only=True).fillna(0.0)
        climatology[city][m] = (m_mean, m_std)

def generate_for_city(city_df, start_date, end_date, n_samples=1):
    last_row = city_df.iloc[-1]
    last_lat, last_lon = last_row["LATITUDE"], last_row["LONGITUDE"]
    last_date = city_df["DATE"].max()
    if N_DAYS_MAX is not None:
        days = N_DAYS_MAX
        end_date = last_date + pd.Timedelta(days=days)
    else:
        end_date = pd.to_datetime(end_date)
        days = (end_date - last_date).days
        if days <= 0:
            return []  

    ref = city_df.tail(30)[numeric_cols].mean(numeric_only=True).fillna(0.0)
    ref_std = city_df.tail(30)[numeric_cols].std(numeric_only=True).fillna(0.0)

    out_rows = []
    for sample_idx in range(n_samples):
        for d in range(1, days + 1):
            dt = last_date + pd.Timedelta(days=d)
            month = dt.month

            if city in climatology:
                clim_mean, clim_std = climatology[city].get(month, (ref, ref_std))
            else:
                clim_mean, clim_std = (ref, ref_std)

            h = d / max(days, 1)
            w_ref = max(0.0, 1.0 - h**1.0) 
            w_clim = 1.0 - w_ref

            base = w_ref * ref + w_clim * clim_mean

            combined_std = (ref_std * (1 - h) + clim_std * h).fillna(0.0)
            noise = np.random.normal(0, combined_std * 0.6)

            row_vals = base + noise
            row_vals["T2M_MAX"] = np.clip(row_vals.get("T2M_MAX", row_vals.get("T2M")), -50, 60)
            row_vals["T2M_MIN"] = np.clip(row_vals.get("T2M_MIN", row_vals.get("T2M")), -60, row_vals["T2M_MAX"])
            row_vals["T2M"] = (row_vals.get("T2M_MAX", row_vals["T2M"]) + row_vals.get("T2M_MIN", row_vals["T2M"])) / 2.0
            row_vals["RH2M"] = np.clip(row_vals.get("RH2M", 50.0), 0, 100)
            row_vals["WS2M"] = np.clip(row_vals.get("WS2M", 0.0), 0, 40)
            row_vals["ALLSKY_SFC_UV_INDEX"] = np.clip(row_vals.get("ALLSKY_SFC_UV_INDEX", 0.0), 0, 15)
            epsilon = 1e-2
            row_vals["CLOUD_OD"] = np.clip(row_vals.get("CLOUD_OD", 0.0), 0, 1.0 - epsilon)
            row_vals["PRECTOTCORR"] = max(0.0, row_vals.get("PRECTOTCORR", 0.0))
            row_vals["PS"] = np.clip(row_vals.get("PS", 1013.0), 800, 1080)
            row_vals["WD2M"] = np.mod(row_vals.get("WD2M", 0.0), 360)
            row_vals["ALLSKY_KT"] = np.clip(row_vals.get("ALLSKY_KT", 0.0), 0, 1)

            out = {
                "DATE": dt,
                "YEAR": dt.year,
                "MO": dt.month,
                "DY": dt.day,
                "CITY": city,
                "LATITUDE": last_lat,
                "LONGITUDE": last_lon
            }
            for col in numeric_cols:
                out[col] = float(row_vals.get(col, np.nan))
            out_rows.append(out)
    return out_rows

all_future_rows = []
for city in tqdm(cities, desc="Generating per-city"):
    city_df = df[df["CITY"] == city].sort_values("DATE")
    if city_df.empty:
        continue
    rows = generate_for_city(city_df, start_date=None, end_date=END_DATE, n_samples=N_SAMPLES)
    all_future_rows.extend(rows)

future_df = pd.DataFrame(all_future_rows)

if future_df.empty:
    print("No future rows generated. Check dates and data.")
else:
    original = df.copy()
    cols_order = ["YEAR", "MO", "DY"] + numeric_cols + ["CITY", "LATITUDE", "LONGITUDE", "DATE"]
    future_save = future_df[cols_order]
    combined = pd.concat([original[cols_order], future_save], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_CSV, index=False)
    print("Saved extended dataset to:", OUTPUT_CSV)
