from ml.MLModel import MLModel
from ml.DataPreparer import DataPreparer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#df -> dataframe, nasz domyślny to ten SleepHealth
#target_col -> co chcemy przewidzieć, musi być typu liczbowego
#feature_cols -> kolumny na podstawwie których chcemy przewidzieć, mogą być dowolne ale nie mogą zawierać target_col (nawet jak zawierają to i tak zostanie to usnięte w DataPreperer)
#ewentualnie możemy wpisać size czyli w jakiej proporcji dzielimy nasze wiersze (treningowe/testowe)
#gdy to wpiszemy to automatycznie stworzy się cała klasa, łącznie z podzieleniem na dane treningowe i testowe
#możemy z niej wydobyć np MSE albo coefficients i dowiedzieć się czy dana zmienna wpływa dodatnio czy ujemnie na wynik


class LinearRegressionModel(MLModel):
    # Atrybuty do przechowywania danych testowych i wyników
    X_test = None
    y_test = None
    feature_names = None
    coefficients = None

    def __init__(self, df=None, target_col=None, feature_cols=None, test_size=0.2): #inicjujemy obiekt LinearRegressionModel
        #czyli wstawiamy dataframe, co chcemy przewidzieć i na podstawie czego chcemy to przewidzieć oraz jak duża ma być próbka testowa
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
            self.feature_names = X_train.columns.tolist()
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

        for name, coef in zip(self.feature_names, self.coefficients): #funckja zip łączy elementy z dwóch list w krotki parując po  tym samym indeksie

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