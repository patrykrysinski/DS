import pandas as pd
import random
from datetime import datetime, timedelta

# Kategorie i przykładowe sklepy
categories = {
    "jedzenie": ["Biedronka", "Lidl", "Żabka", "McDonald's", "KFC"],
    "transport": ["PKP", "Bolt", "Uber", "Orlen", "Shell"],
    "rozrywka": ["Netflix", "Spotify", "Cinema City", "Empik"],
    "rachunki": ["Energa", "UPC", "Orange", "Vectra"],
    "inne": ["Allegro", "Amazon", "Rossmann"],
}

# Ustawienia
num_rows = 100
start_date = datetime(2025, 1, 1)
data = []

for _ in range(num_rows):
    category = random.choice(list(categories.keys()))
    merchant = random.choice(categories[category])
    amount = round(random.uniform(10, 500), 2)
    date = start_date + timedelta(days=random.randint(0, 300))
    data.append(
        {
            "Date": date.strftime("%Y-%m-%d"),
            "Merchant": merchant,
            "Category": category,
            "Amount": amount,
        }
    )

# Tworzenie DataFrame
df = pd.DataFrame(data)

# Zapis do pliku CSV
df.to_csv("transactions.csv", index=False, encoding="utf-8")

print("✅ Plik data/transactions.csv został utworzony!")
print(df.head(10))
