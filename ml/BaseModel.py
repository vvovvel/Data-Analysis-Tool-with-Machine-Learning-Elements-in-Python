import pandas as pd
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train=None):
        try:
            if y_train is not None:
                if isinstance(y_train, pd.Series):
                    y_train = y_train.values
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train)
        except Exception as e:
            print(f"[ERROR] Błąd podczas trenowania: {e}")
            raise e

    def predict(self, X_test):
        try:
            return self.model.predict(X_test)
        except Exception as e:
            print(f"[ERROR] Błąd podczas predykcji: {e}")
            raise e

    @abstractmethod
    def evaluate(self): #wymagana metoda w klasach bazowych, w każdej inaczej implementowana
        pass

    @abstractmethod
    def _draw_plot_content(self, plt): #specyficzna metoda rysowania dla każdego modelu
        pass

    def plot(self, filename: str = "model_plot.png"): #metoda zawiera wspólną logikę wszystkich wykresów
        try:
            os.makedirs('outputs', exist_ok=True) #sprawdzanie folderu outputs
            plt.figure(figsize=(10, 6)) #wspólna wielkość

            #wywołanie metody specyficznej
            self._draw_plot_content(plt)

            #wspólne elementy stylu
            plt.grid(True, linestyle='--', alpha=0.7)

            #zapis i czyszczenie
            plot_path = os.path.join('outputs', filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Wykres modelu zapisany: {plot_path}")

        except ValueError as e:
            # Wychwycenie błędów walidacji (np. zła liczba cech) rzuconych z _draw_plot_content
            print(f"Błąd walidacji danych do wykresu: {e}")
            plt.close()  #upewnienie się, że figura jest zamknięta
        except Exception as e:
            print(f"Błąd podczas generowania wykresu: {e}")
            plt.close()
            raise e