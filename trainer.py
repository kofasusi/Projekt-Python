# trainer.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# trainer.py
def train_model(model, X_scaled, y, epochs=10):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_scaled, y, epochs=epochs, verbose=1)
    return model, history


def plot_predictions(model, X, y_scaled, y_scaler, sequence_length=30):
    predictions = model.predict(X)
    predictions_rescaled = y_scaler.inverse_transform(predictions)
    y_rescaled = y_scaler.inverse_transform(y_scaled)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(y_rescaled, label="Rzeczywiste ceny")
    plt.plot(predictions_rescaled, label="Predykcje")
    plt.title("Predykcja cen akcji")
    plt.xlabel("Dzień")
    plt.ylabel("Cena")
    plt.legend()
    plt.show()
