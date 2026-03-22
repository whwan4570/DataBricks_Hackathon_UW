# -*- coding: utf-8 -*-
"""
두 데이터 병합: NOAA 일일 날씨 + RMA 카운티 수확량
- 날씨: 연도별 집계(평균 기온, 강수량 합계)
- 수확량: State, County, Year, Commodity, Yield
- 병합 키: 연도 (Yield Year = date의 연도)
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
WEATHER_CSV = BASE / "Explore_rearc_daily_weather_observations_noaa_esg_noaa_ghcn_noaa_ghcn_daily_2026_03_07_02_53_30.csv"
YIELDS_CSV = BASE / "RMACountyYieldsReport-399.csv"
OUTPUT_CSV = BASE / "merged_weather_yields.csv"

def load_weather(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    # temp는 0.1°C 단위로 저장된 것으로 가정 (NOAA GHCN 관례)
    for col in ["temp_max", "temp_min", "precipitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def aggregate_weather_by_year(df):
    agg = df.groupby("year").agg(
        avg_temp_max=("temp_max", "mean"),
        avg_temp_min=("temp_min", "mean"),
        total_precipitation=("precipitation", "sum"),
        station_count=("station", "nunique"),
    ).reset_index()
    # 0.1°C -> °C
    agg["avg_temp_max_c"] = (agg["avg_temp_max"] / 10).round(2)
    agg["avg_temp_min_c"] = (agg["avg_temp_min"] / 10).round(2)
    agg["total_precipitation_mm"] = agg["total_precipitation"].round(2)
    return agg[["year", "avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm", "station_count"]]

def load_yields(path):
    df = pd.read_csv(path)
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    return df

def main():
    print("Loading weather data...")
    weather = load_weather(WEATHER_CSV)
    print(f"  Rows: {len(weather)}, Date range: {weather['date'].min()} ~ {weather['date'].max()}")

    print("Aggregating weather by year...")
    weather_year = aggregate_weather_by_year(weather)
    print(weather_year.to_string())

    print("\nLoading yields data...")
    yields = load_yields(YIELDS_CSV)
    yields = yields.rename(columns={"Yield Year": "year"})
    print(f"  Rows: {len(yields)}, Years: {yields['year'].min()} ~ {yields['year'].max()}")

    print("\nMerging on year...")
    merged = yields.merge(weather_year, on="year", how="left")
    merged = merged.sort_values(["State Name", "County Name", "Commodity Name", "year"])

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Merged rows: {len(merged)}")
    print("\nFirst rows:")
    print(merged.head(10).to_string())

if __name__ == "__main__":
    main()
