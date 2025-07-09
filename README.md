Predykcja cen akcji z uzyciem TensorFlow i danych historycznych\
Cel Projektu\
Aplikacja w języku Python która analizuje historyczne dane z giełdy, a następnie próbuje przewidzieć nowe dane.\
Rezultatem będzie działający program oparty o framework TensorFlow, który wykorzystuje prosty model sieci neuronowej do prognozowania. 
Opis funkcjonalności projektu
Pobieranie historycznych danych akcji z API (np. Yahoo Finance przez yfinance)

Przetwarzanie danych (np. wyciąganie wartości zamknięcia i normalizacja)

Uczenie modelu predykcyjnego (np. LSTM lub Dense NN)

Prognozowanie przyszłych cen akcji

Obsługa błędów API i danych

Kod rozdzielony na moduły (np. data_loader, model, trainer, main)

Użycie generatorów do przetwarzania danych

Prosty test jednostkowy do sprawdzenia działania modelu


Instalacja:

```bash
pip install tensorflow numpy pandas yfinance scikit-learn matplotlib
```
Uruchomienie:

```bash
python main.py
```

Struktura projektu
data_loader.py — pobieranie i przygotowanie danych

model.py — definicja modelu sieci neuronowej

trainer.py — trenowanie modelu i wykresy

utils.py — funkcje pomocnicze (zapisywanie modelu, skalera)

main.py — skrypt uruchamiający cały proces

Możliwe usprawnienia
Dodanie interfejsu użytkownika

Eksperymenty z różnymi modelami

Automatyzacja pobierania danych i aktualizacji modelu

