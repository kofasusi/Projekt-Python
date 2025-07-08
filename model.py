# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_dense_model(input_shape):
    """
    Tworzy prosty model Dense (siec neuronowa w pełni połączona).
    :param input_shape: krotka np. (30,) jeśli mamy sekwencje 30 dni
    :return: gotowy model
    """
    model = Sequential()
    
    # Warstwa wejściowa
    model.add(Input(shape=input_shape))

    # Ukryte warstwy
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Warstwa wyjściowa (jedna wartość – cena)
    model.add(Dense(1))

    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # średni błąd kwadratowy – dobry dla regresji
        metrics=['mae']  # średni błąd absolutny – czytelniejszy
    )

    return model
