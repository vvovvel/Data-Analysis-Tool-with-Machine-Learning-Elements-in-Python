import pandas as pd
from ml.MLModel import MLModel
from ml.DataPreparer import DataPreparer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#nazwa KNN -> K-nearest-neighbors -> model znajduje K sąsiadów i jest przypisywany do klasy, która jest najczęściej reprezentowana wśród nich

#w tym modelu target_col musi zawierać zmienne kategoryczne czyli np occupation, bmi category, sleep disorder
#inicjujemy model poprzez wpisanie df, target_col oraz feature_cols które w domyśle są wszystkimi kolumnami prócz targetu oraz id
#następnie możemy dostać się do accuracy czyli do proporcji poprawnych dopasowań do wszystkich dopasowań

class KNNClassifierModel(MLModel):
    # Atrybuty do przechowywania danych testowych i parametrów
    X_test = None
    y_test = None
    n_neighbors = None
    feature_names = None  # Dodany atrybut

    def __init__(self, df=None, target_col=None, feature_cols=None, n_neighbors=5, test_size=0.2):

        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors)) #model importowany z sklearn
        self.n_neighbors = n_neighbors

        if df is not None and target_col is not None:
            preparer = DataPreparer(df, id_col='Person ID')

            # Używamy preparer.prepare_classification (który skaluje dane)
            X, y, X_train, self.X_test, y_train, self.y_test = preparer.prepare_classification(
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_size
            )

            # Trenujemy model
            self.train(X_train, y_train)

            # Zapisujemy użyte nazwy cech (jest to DataFrame przed skalowaniem)
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
            # W DataPreparer po skalowaniu X_train jest już NumPy array, więc używamy obiektu X przed skalowaniem
            else:
                self.feature_names = X.columns.tolist()  # X jest obiektem przed skalowaniem/podziałem

    def evaluate(self):
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Model nie został poprawnie zainicjowany z danymi.")

            y_pred = self.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return accuracy

        except Exception as e:
            return {"Error": f"Nie można policzyć Accuracy: {e}"}

