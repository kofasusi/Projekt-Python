
Predykcja cen akcji z wykorzystaniem sieci neuronowej w TensorFlow.

Cel projektu:

Stworzenie programu, który pobiera historyczne dane giełdowe (np. AAPL) i przewiduje przyszłe ceny na podstawie wcześniejszych wartości. Celem jest nauka obróbki danych finansowych, trenowania modelu oraz interpretacji wyników.

Opis funkcjonalności:

Pobieranie danych z internetu (yfinance)

Przygotowanie danych (tworzenie sekwencji czasowych)

Skalowanie danych wejściowych

Budowa i trening modelu neuronowego

Wykres predykcji vs rzeczywiste dane


Instalacja:

```bash
pip install -r requirements.txt
```
Uruchomienie:

```bash
python main.py
```

Technologie użyte w projekcie:

-Python 3.9+

-TensorFlow / Keras

-Pandas, NumPy

-scikit-learn

-yfinance

-Matplotlib


Wymagania funkcjonalne:

Program pobiera dane giełdowe.

Trenuje model na podstawie danych historycznych.

Wyświetla wykres predykcji.

Wymagania niefunkcjonalne:

Program ma działać lokalnie (Python CLI).

Kod zorganizowany modułowo (pliki: model.py, trainer.py, data_loader.py, main.py).

Używa bibliotek zgodnych z Python ≥ 3.9.

Interfejs użytkownika:

Użytkownik uruchamia program przez main.py — tekstowy interfejs CLI, wykres wyświetlany graficznie.



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

