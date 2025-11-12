from ml.MLModel import MLModel
import numpy as np
from ml.DataPreparer import DataPreparer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClusteringModel(MLModel):
    X_scaled = None
    n_clusters = None

    def __init__(self, df=None, feature_cols=None, n_clusters=3, random_state=42):

        super().__init__(KMeans(n_clusters=n_clusters, random_state=random_state)) #model importowany z sklearn
        self.n_clusters = n_clusters  # Zapisujemy liczbę klastrów

        # 2. Logika przygotowania i trenowania
        if df is not None:
            preparer = DataPreparer(df, id_col='Person ID')

            # Klasteryzacja NIE MA TARGET_COL, używamy metody prepare_clustering
            # prepare_clustering automatycznie koduje kategorie i SKALUJE cały zbiór.
            X_scaled = preparer.prepare_clustering(feature_cols=feature_cols)
            self.X_scaled = X_scaled  # Zapisujemy skalowane dane do ewaluacji

            # Trenujemy model. W Klasteryzacji trenujemy na X bez Y
            self.train(X_scaled)

    # 3. Uproszczona metoda evaluate (używa wewnętrznych, skalowanych danych)
    def evaluate(self):
        try:
            X_data = self.X_scaled

            if X_data is None:
                raise ValueError("Model nie został poprawnie zainicjowany z danymi.")

            labels = self.predict(X_data)

            # Wymagane są minimum 2 unikalne klastry do obliczenia Silhouette Score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_data, labels)
                return {"Silhouette Score": round(score, 4)}
            else:
                return {"Info": f"Za mała liczba klastrów ({self.n_clusters}) do oceny silhouette_score"}

        except Exception as e:
            return {"Error": f"Nie można policzyć silhouette_score: {e}"}