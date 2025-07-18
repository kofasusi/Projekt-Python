# main.py

from data_loader import download_stock_data, prepare_sequences
from model import build_dense_model
from trainer import train_model, plot_predictions
from sklearn.preprocessing import MinMaxScaler
from dataset import StockDataset

def main():
    # 1. Pobierz dane
    data = download_stock_data("AAPL", period="100d")
    if data.empty:
        print("Brak danych, kończę program.")
        return

    # 2. Przygotuj dane

    dataset = StockDataset("AAPL", data['Close'].values, sequence_length=30)
    X, y = dataset.get_features(), dataset.get_targets()
    print(dataset.summary())
    X = X.reshape(X.shape[0], X.shape[1])  # spłaszcz jeśli X ma dodatkowy wymiar
    # 3. Skaluj dane
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

    # 4. Zbuduj model
    model = build_dense_model(input_shape=(X_scaled.shape[1],))

    # 5. Trenuj model
    model, history = train_model(model, X_scaled, y_scaled, epochs=10)

    # 6. Wyświetl wykres predykcji
    plot_predictions(model, X_scaled, y_scaled, y_scaler)

if __name__ == "__main__":
    main()
