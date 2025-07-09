import os
import pickle

def save_model(model, path="model.h5"):
    model.save(path)
    print(f"Model zapisany w {path}")

def load_model(tf, path="model.h5"):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    else:
        print("Model nie istnieje.")
        return None

def save_scaler(scaler, path="scaler.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler zapisany w {path}")

def load_scaler(path="scaler.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Scaler nie istnieje.")
        return None
