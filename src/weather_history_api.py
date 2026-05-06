import requests

# Your coordinates
latitude = 50.496406
longitude = 7.895274

# Historical weather API URL
url = "https://archive-api.open-meteo.com/v1/archive"

# Dates we want to check
dates_to_check = ["2019-03-08", "2018-05-06"]

print("Historical Weather Data")
print("-----------------------")

for date in dates_to_check:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }

    response = requests.get(url, params=params)
    data = response.json()

    daily = data["daily"]

    print(
        f"{daily['time'][0]} | "
        f"Max: {daily['temperature_2m_max'][0]}°C | "
        f"Min: {daily['temperature_2m_min'][0]}°C | "
        f"Rain: {daily['precipitation_sum'][0]} mm"
    )

print("\nComparison Insight")
print("------------------")

today_2018 = 21.4
today_2026_forecast = 11.8  # from your forecast

diff = today_2026_forecast - today_2018

print(f"Difference vs 2018: {diff:.1f}°C")
