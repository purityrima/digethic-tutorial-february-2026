import requests

# Your coordinates
latitude = 50.496406
longitude = 7.895274

# API URL
url = "https://api.open-meteo.com/v1/forecast"

# Parameters
params = {
    "latitude": latitude,
    "longitude": longitude,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    "forecast_days": 5,
    "timezone": "auto",
}

# Request
response = requests.get(url, params=params)

# Convert to JSON
data = response.json()

# Print result
# print(data)
daily = data["daily"]

dates = daily["time"]
max_temps = daily["temperature_2m_max"]
min_temps = daily["temperature_2m_min"]
rain = daily["precipitation_sum"]

print("5-Day Weather Forecast")
print("----------------------")

for i in range(len(dates)):
    print(
        f"{dates[i]} | "
        f"Max: {max_temps[i]}°C | "
        f"Min: {min_temps[i]}°C | "
        f"Rain: {rain[i]} mm"
    )
