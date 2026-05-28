# Import Libraries
from urllib import request, parse, error
import json
import csv
from datetime import date


target_info = {"name": "Seoul (fallback)",
               "latitude": 37.5665,
               "longitude": 126.9780,
               "country_code": "KR",
               "timezone": "Asia/Seoul"
               }


def fetch_daily(lat, lon, start_date, end_date, timezone,
                daily_vars=("temperature_2m_max", "temperature_2m_min",
                            "temperature_2m_mean", "precipitation_sum", "windspeed_10m_max")):
    base = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone,
    }
    url = f"{base}?{parse.urlencode(params)}"
    with request.urlopen(url, timeout=20) as resp:
        data = json.load(resp)
    daily = data.get("daily")
    if not daily:
        raise RuntimeError("Error has been occurred")
    return daily


def save_daily_to_csv(daily: dict, out_path: str):
    header = ["time"] + [k for k in daily.keys() if k != "time"]
    rows = []
    times = daily["time"]
    for i in range(len(times)):
        row = [times[i]]
        for k in header[1:]:
            row.append(daily.get(k, [""] * len(times))[i])
        rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    # Target location
    lat, lon = target_info["latitude"], target_info["longitude"]
    timezone = target_info["timezone"]

    # Target period to collect
    start, end = "2024-01-01", "2024-12-31"

    # Request data via API and save the result
    daily = fetch_daily(lat, lon, start, end, timezone)
    out = f"6-1_DataCollection/seoul_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    save_daily_to_csv(daily, out)
    print("Requested Data Has Been Saved")
