# test_data_loader.py

from data_loader import download_stock_data, prepare_sequences

# Test 1 – pobieranie danych
data = download_stock_data("AAPL", period="3mo")  # 3 miesiące danych

if not data.empty:
    print("✅ Dane pobrane!")
    print(data.head())
else:
    print("❌ Coś poszło nie tak.")

# Test 2 – przygotowanie sekwencji
X, y = prepare_sequences(data, sequence_length=10)
print(f"📊 Liczba sekwencji: {len(X)}")
print(f"Przykład sekwencji X[0]: {X[0]}")
print(f"Cel y[0]: {y[0]}")
