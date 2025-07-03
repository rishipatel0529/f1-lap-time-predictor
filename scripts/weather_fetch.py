# scripts/weather_fetch.py
import os
import sys

import requests
from dotenv import load_dotenv

# Load key, silently
load_dotenv("config/.env")
api_key = os.getenv("OWM_API_KEY")
if not api_key:
    print("OWM_API_KEY not set!", file=sys.stderr)
    sys.exit(1)

# Build request
url = "https://api.openweathermap.org/data/3.0/onecall"
params = {"lat": 52.0786, "lon": -1.0169, "units": "metric", "appid": api_key}

# Fetch
resp = requests.get(url, params=params)
if resp.status_code != 200:
    print(f"HTTP {resp.status_code}", file=sys.stderr)
    print(resp.text, file=sys.stderr)
    sys.exit(1)

print(resp.text)
