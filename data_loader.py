# data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Pobiera dane giełdowe z Yahoo Finance.
    :param ticker: symbol akcji (np. AAPL)
    :param period: okres (np. 1y = 1 rok)
    :return: DataFrame z danymi
    """
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError("Brak danych. Sprawdź symbol akcji.")
        return data[['Close']]
    except Exception as e:
        print(f"Błąd pobierania danych: {e}")
        return pd.DataFrame()

def prepare_sequences(data: pd.DataFrame, sequence_length: int = 30):
    """
    Tworzy sekwencje danych do trenowania modelu.
    :param data: DataFrame z kolumną 'Close'
    :param sequence_length: długość jednej sekwencji (np. 30 dni)
    :return: X (wejście), y (cel)
    """
    close_prices = data['Close'].values
    X, y = [], []

    for i in range(len(close_prices) - sequence_length):
        seq = close_prices[i:i + sequence_length]
        target = close_prices[i + sequence_length]
        X.append(seq)
        y.append(target)

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1])  # wymuszenie 2D dla Dense modelu
    y = np.array(y)
    return X, y
