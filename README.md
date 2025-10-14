# NZ Daily Weather Data Script

This project provides a Python script that fetches daily weather data for major New Zealand cities using the free Open-Meteo API (no API key required), and writes a cleaned CSV dataset.

## Quick Start

1. Create a virtual environment (optional but recommended)
   - PowerShell:
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`

2. Install dependencies
   - `pip install -r requirements.txt`

3. Run the script
   - Today’s data: `python weather_nz.py`
   - Specific date: `python weather_nz.py --date 2025-01-31`
   - Date range: `python weather_nz.py --start 2025-01-01 --end 2025-01-31`
   - Specific cities: `python weather_nz.py --cities Auckland Wellington`
   - Custom output: `python weather_nz.py --out data/daily_weather_nz.csv`

The output CSV defaults to `data/daily_weather_nz.csv` and includes columns like `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum`, `windspeed_10m_max`, and more.

## Notes

- Timezone is set to `Pacific/Auckland`.
- The script applies basic retries with exponential backoff and a small delay between city requests.
- If your environment blocks network access, the script will not be able to fetch data but you can still review the code and usage.

## Cities Covered

Auckland, Wellington, Christchurch, Hamilton, Tauranga, Dunedin, Palmerston North, Napier, Nelson, Rotorua, New Plymouth, Invercargill, Whangarei.

