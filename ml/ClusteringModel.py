import numpy as np

from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Algorytm przesuwa "środki" klastrów (centroidy), aż każdy punkt danych będzie najbliżej swojego centroidu.

class ClusteringModel(BaseModel):

    def __init__(self, df=None, id_col=None ,feature_cols=None, n_clusters=3, random_state=42):
        self.id_col = id_col
        self.X_scaled = None
        self.feature_cols = None
        self.n_clusters = None

        super().__init__(KMeans(n_clusters=n_clusters, random_state=random_state)) #model importowany z sklearn
        self.n_clusters = n_clusters #liczba klastrów tzn grup

        if df is not None:
            preparer = DataPreparer(df, id_col)

            X_scaled = preparer.prepare_clustering(feature_cols=feature_cols)  # przygotowywanie danych pod KMeans
            self.X_scaled = X_scaled
            self.feature_cols = X_scaled.columns.tolist()  # Zapis nazw kolumn cech

            self.train(X_scaled)  # trenowanie model


    def evaluate(self): #shilhouette_score - współczynnik sylwetkowy -> mierzy jakość separacji i spójność klastrów
        #ocenia jak bardzo punkt jest podobny do swojego klastra i jak bardzo różny jest od innych klastrów. wartości od -1 do 1
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

    # specyficzne rysowanie wykresu

    def _draw_plot_content(self, plt):

        # 1. Walidacja danych i cech (zmieniono 'print' na 'raise ValueError' by obsłużył to BaseModel)
        # Do narysowania modelu musimy uprzednio zainicjować i wytrenować model
        if self.X_scaled is None:
            raise ValueError("Model musi zostać zainicjowany z danymi.")

        if self.feature_cols is None:
            raise ValueError("Brak nazw cech (feature_cols).")

        # Walidacja: wymagamy dokładnie dwóch cech
        if len(self.feature_cols) != 2:
            raise ValueError(f"Wizualizacja wymaga dokładnie 2 cech. Znaleziono: {len(self.feature_cols)}.")

        # 2. Pobieranie danych i nazw z atrybutów self
        feature_name_1 = self.feature_cols[0]  # Pierwsza cecha
        feature_name_2 = self.feature_cols[1]  # Druga cecha

        # Dane do rysowania
        X_data = self.X_scaled.values
        labels = self.predict(self.X_scaled)  # trenowanie modelu
        n_unique_clusters = len(np.unique(
            labels))  # zazwyczaj n_unique_clusters == n_clusters ale gdy model jest mały lub niesymetryczny może być <

        # Wymagane są klastry do narysowania
        if n_unique_clusters < 1:
            raise ValueError("Nie znaleziono klastrów do wizualizacji.")

        # --- Rysowanie ---

        cmap = plt.get_cmap('Dark2', n_unique_clusters)  # Colormapa dla klastrów

        # Wykres rozrzutu (Punkty Danych)
        for i in range(n_unique_clusters):  # rysujemy dla każdego klastra punkty, które się w nim znajdują
            mask = (labels == i)  # maska, która da nam tylko punkty, które są w danym klastrze
            plt.scatter(X_data[mask, 0], X_data[mask, 1],  # rysowanie każdego punktu z klastra
                        label=f'Klaster {i}',
                        color=cmap(i),  # dobieranie odpowiedniego koloru
                        alpha=0.6,
                        edgecolors='w',  # obwódki
                        linewidths=0.5)

        if self.model.cluster_centers_.any():  # centroidy, środki klastrów
            centers_plot = self.model.cluster_centers_  # atrybut KMeans w postaci tablicy Numpy rozmiaru n_clusters x 2 (bo n_features = 2) w wypelniona współrzędnymi centroidów

            plt.scatter(centers_plot[:, 0], centers_plot[:, 1],  # rysujemy centroidy
                        marker='X', s=200,
                        color='black',
                        label='Centroidy',
                        edgecolors='k',
                        linewidths=1.5)

        plt.title(f'Wizualizacja klasteryzacji: {feature_name_1} vs {feature_name_2}', fontsize=14)
        plt.xlabel(feature_name_1, fontsize=12)
        plt.ylabel(feature_name_2, fontsize=12)
        plt.legend(loc='best')  # lokalizacja dynamiczna, w najlepszym możliwym miejscu

    # wywołanie metody bazowej, która zawiera w sobie powtarzające się u wszystkich modeli fragmetny kodu

    def plot(self, filename: str = "clustering_plot.png"):
        super().plot(filename=filename)