# dataset.py

import numpy as np

class StockDataset:
    def __init__(self, ticker: str, close_prices: np.ndarray, sequence_length: int = 30):
        self.ticker = ticker
        self.close_prices = close_prices
        self.sequence_length = sequence_length
        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.close_prices) - self.sequence_length):
            X.append(self.close_prices[i:i+self.sequence_length])
            y.append(self.close_prices[i+self.sequence_length])
        return np.array(X), np.array(y)

    def get_features(self):
        return self.X

    def get_targets(self):
        return self.y

    def summary(self):
        return {
            "ticker": self.ticker,
            "num_samples": len(self.X),
            "sequence_length": self.sequence_length
        }
