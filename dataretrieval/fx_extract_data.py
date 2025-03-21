import requests
import pandas as pd
import datetime

# 📌 API instellingen (Frankfurter API zonder API-key)
BASE_CURRENCY = "USD"
TARGET_CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CHF"]
API_URL = f"https://api.frankfurter.app/latest?from={BASE_CURRENCY}"

# 📌 Data ophalen
response = requests.get(API_URL)
data = response.json()

# 📌 Data verwerken en opslaan
if "rates" in data:
    forex_data = {
        "date": data["date"],
        "base_currency": BASE_CURRENCY
    }

    for currency in TARGET_CURRENCIES:
        forex_data[currency] = data["rates"].get(currency, None)

    # Dataframe maken en opslaan als CSV
    df = pd.DataFrame([forex_data])
    df.to_csv("datafiles/forex_rates.csv", index=False)

    print("✅ Forex data succesvol opgehaald en opgeslagen in 'data/forex_rates.csv'.")

else:
    print(f"❌ Fout: Kon geen forex data ophalen. Response: {data}")
