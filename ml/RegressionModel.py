from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#df -> dataframe, nasz domyślny to ten SleepHealth
#target_col -> co chcemy przewidzieć, musi być typu liczbowego
#feature_cols -> kolumny na podstawwie których chcemy przewidzieć, mogą być dowolne ale nie mogą zawierać target_col (nawet jak zawierają to i tak zostanie to usnięte w DataPreperer)
#ewentualnie możemy wpisać size czyli w jakiej proporcji dzielimy nasze wiersze (treningowe/testowe)
#gdy to wpiszemy to automatycznie stworzy się cała klasa, łącznie z podzieleniem na dane treningowe i testowe
#możemy z niej wydobyć np MSE albo coefficients i dowiedzieć się czy dana zmienna wpływa dodatnio czy ujemnie na wynik


class RegressionModel(BaseModel):

    #konstruktor
    def __init__(self, df=None, target_col=None, feature_cols=None, test_size=0.2):
        #zapisujemy tylko niektóre zmienne, które będą nam potrzebne w innych metodach np evaluate, plot

        self.X_test = None
        self.y_test = None
        self.feature_cols = None
        self.target_col = None
        self.coefficients = None

        #konstruktor klasy nadrzędnej
        super().__init__(LinearRegression()) #odnosi się do klasy nadrzędnej czyli MLModel i wywołuje tamten konstruktor z modelem LinearRegression zaimportowanym z sklearn

        if df is not None and target_col is not None: #sprawdzamy oczywiście czy nie ma bzdurnych danych, zakładamy już że df jest po loadowaniu, walidacji itd
            preparer = DataPreparer(df, id_col='Person ID') #tworzymy obiekt DataPreperer

            X, y, X_train, self.X_test, y_train, self.y_test = preparer.prepare_data_regression(
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_size
            ) #wywołując odpowiednią metodę

            # Trenujemy model
            self.train(X_train, y_train)

            # Zapisujemy wyniki analizy po trenowaniu
            self.target_col = target_col
            self.feature_cols = X_train.columns.tolist()
            self.coefficients = self.model.coef_

    def evaluate(self):
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Model nie został poprawnie zainicjowany z danymi.")

            y_pred = self.predict(self.X_test) #predict znajduje się w MLModel ale MLModel predict odwołuje się do modelu samego w sobie czyli w tym przypadku LinearRegression
            mse = mean_squared_error(self.y_test, y_pred)
            return mse
        except Exception as e:
            return {"Error": f"Nie można policzyć MSE: {e}"}

    def get_analysis_summary(self):
        if self.coefficients is None:
            return "Model nie został jeszcze wytrenowany."

        summary = {"Współczynniki Regresji": {}}

        for name, coef in zip(self.feature_cols, self.coefficients): #funckja zip łączy elementy z dwóch list w krotki parując po  tym samym indeksie

            if coef > 0:
                impact = "Dodatni"
            elif coef < 0:
                impact = "Ujemny"
            else:
                impact = "Brak wpływu"

            summary["Współczynniki Regresji"][name] = {
                "Wpływ": impact,
                "Wartość": round(coef, 4)
            }

        return summary

    def plot(self, filename: str = "regresja_plot.png"):

        # 1. Walidacja danych i cech
        # Do narysowania modelu musimy uprzednio zainicjować i wytrenować model
        if self.X_test is None or self.y_test is None:
            print("Model musi zostać zainicjowany")
            return

        if self.feature_cols is None:
            print("Brak nazw cech (feature_cols).")
            return

        if len(self.feature_cols) != 1:
            print(f"Zbyt duża ilośc cech")
            return

        try:
            os.makedirs('outputs', exist_ok=True)  # Zabezpieczenie folderu

            # 2. Pobieranie danych i nazw z atrybutów self
            feature_name = self.feature_cols[0]  # JEDYNA cecha
            target_name = self.target_col  # Nazwa Targetu

            # 3. Przygotowanie danych osi
            # Wybór JEDYNEJ kolumny z X_test i konwersja do wektora 1D
            x_data = self.X_test[feature_name].values.flatten()
            y_test_data = self.y_test.values.flatten()

            # 4. Obliczenie predykcji linii regresji
            y_pred = self.predict(self.X_test).flatten()

            # --- Rysowanie ---
            plt.figure(figsize=(10, 6))

            # Wykres rozrzutu (Punkty Danych)
            plt.scatter(x_data, y_test_data, color='#3498db', alpha=0.6, label='Dane Testowe')

            # Linia regresji (Predykcja)
            # Musimy posortować dane X, aby linia regresji była narysowana poprawnie
            sort_idx = x_data.argsort()
            plt.plot(x_data[sort_idx], y_pred[sort_idx], color='#e74c3c', linewidth=3, label='Linia Regresji')

            plt.title(f'Regresja Liniowa: {target_name} vs {feature_name}', fontsize=14)
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel(target_name, fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # --- Zapis ---
            plot_path = os.path.join('outputs', filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Wykres regresji zapisany: {plot_path}")

        except Exception as e:
            print(f"Błąd podczas generowania wykresu: {e}")