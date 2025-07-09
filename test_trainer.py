# test_trainer.py

from data_loader import download_stock_data, prepare_sequences
from model import build_dense_model
from trainer import train_model, plot_predictions
from sklearn.preprocessing import MinMaxScaler

# 1. Pobierz dane
data = download_stock_data("AAPL", period="6mo")
if data.empty:
    print("❌ Brak danych, test przerwany.")
    exit()

# Przygotuj dane (X może mieć kształt (samples, 30, 1))
X, y = prepare_sequences(data, sequence_length=30)

# Spłaszcz X do 2D
X = X.reshape(X.shape[0], X.shape[1])  # z (93,30,1) -> (93,30)

# Skaluj dane
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1,1))

# 4. Zbuduj model
model = build_dense_model(input_shape=(X_scaled.shape[1],))

# 5. Trenuj model na przeskalowanych danych y
model, history = train_model(model, X_scaled, y_scaled, epochs=10)

# 6. Wygeneruj wykres predykcji (z deskalowaniem y)
plot_predictions(model, X_scaled, y_scaled, y_scaler)
