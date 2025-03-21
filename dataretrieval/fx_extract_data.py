import requests
import pandas as pd
import datetime

# 📌 API Settings
BASE_CURRENCY = "USD"
TARGET_CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CHF"]
START_DATE = "2023-01-01"  # Adjust the start date as needed
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# 🗓️ Generate a list of dates
date_range = pd.date_range(start=START_DATE, end=END_DATE)

# 📊 Collect data for each date
data_list = []

for date in date_range:
    date_str = date.strftime("%Y-%m-%d")
    API_URL = f"https://api.frankfurter.app/{date_str}?from={BASE_CURRENCY}"
    response = requests.get(API_URL)
    data = response.json()

    if "rates" in data:
        forex_data = {"date": date_str, "base_currency": BASE_CURRENCY}
        for currency in TARGET_CURRENCIES:
            forex_data[currency] = data["rates"].get(currency, None)
        data_list.append(forex_data)
        print(f"✅ Fetched data for {date_str}")
    else:
        print(f"❌ No data for {date_str}")

# 📂 Save data to CSV
df = pd.DataFrame(data_list)
df.to_csv("datafiles/forex_rates.csv", index=False)
print("✅ Historical forex data saved to 'datafiles/forex_rates.csv'")
