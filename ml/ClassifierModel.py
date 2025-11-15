import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#nazwa KNN -> K-nearest-neighbors -> model znajduje K sąsiadów i jest przypisywany do klasy, która jest najczęściej reprezentowana wśród nich

#w tym modelu target_col musi zawierać zmienne kategoryczne czyli np occupation, bmi category, sleep disorder
#inicjujemy model poprzez wpisanie df, target_col oraz feature_cols które w domyśle są wszystkimi kolumnami prócz targetu oraz id
#następnie możemy dostać się do accuracy czyli do proporcji poprawnych dopasowań do wszystkich dopasowań

class ClassifierModel(BaseModel):

    def __init__(self, df=None, target_col=None, feature_cols=None, n_neighbors=5, test_size=0.2):
        self.X_test = None
        self.y_test = None
        self.n_neighbors = None
        self.feature_cols = None
        self.target_col = None

        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors)) #model importowany z sklearn
        self.n_neighbors = n_neighbors
        self.target_col = target_col

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
            self.feature_cols = X_train.columns.tolist()

    def evaluate(self):
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Model nie został poprawnie zainicjowany z danymi.")

            y_pred = self.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return accuracy

        except Exception as e:
            return {"Error": f"Nie można policzyć Accuracy: {e}"}

    def plot(self, filename: str = "classifier_plot.png"):

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
            os.makedirs('outputs', exist_ok=True)

            # Pobieranie nazw
            feature_name = self.feature_cols[0]
            target_name = self.target_col

            # 3. Przygotowanie danych osi
            # Wybór JEDYNEJ kolumny z X_test i konwersja do wektora 1D
            x_data = self.X_test[feature_name].values.flatten()
            y_test_data = np.array(self.y_test).flatten()

            # 4. Obliczenie predykcji
            y_pred = np.array(self.predict(self.X_test)).flatten()

            # Mapowanie klas (np. ['low','medium','high'] -> [0,1,2])
            classes = np.unique(np.concatenate([y_test_data, y_pred])) #połączenie danych testowych i predykowanych w unikalny set
            class_to_num = {cls: i for i, cls in enumerate(classes)} #słownik -> dla każdej unikalnej wartości z classes nadaje jakąś liczbę naturalną
            num_to_class = {i: cls for cls, i in class_to_num.items()} #odwrotne mapowanie: każda liczba daje wartość np string

            # --- Rysowanie ---
            plt.figure(figsize=(10, 6))

            cmap = plt.get_cmap('tab10') #pobiera colormapę z matplotlib -> dobrze rozróznialne 10 kolorów
            class_to_color = {cls: cmap(i % 10) for i, cls in enumerate(classes)} #dla każdej z naszych klas mamy jeden z tych dziesięciu kolorów
            # ale niestety jeśli klas jest więcej to będą się powtarzać. jesteśmy gotowi na to poświęcenie

            # Rysujemy prawdziwe etykiety (filled)
            for cls in classes:
                mask = (y_test_data == cls) #maska, która tworzy dla każdej klasy takie coś
                # [low, high, low, medium] -> [1,0,1,0] dla low. w skrocie -> true gdy ta klasa, false gdy nie ta klasa
                plt.scatter(x_data[mask], np.full(mask.sum(), class_to_num[cls]), #wybieramy tylko wartości dla punktów z tej klasy(argumenty).
                            # następnie tworzymy wartości y (tablica tak duża ile mamy punktów), wszystkie mają wysokość class_to_num bo taką mają wartość zamienione
                            #na liczby naturalne igreki
                            color=class_to_color[cls], #wypełnone
                            alpha=0.6, #półprzezroczyste
                            label=f'True: {cls}')

            # Rysujemy predykcje
            for cls in classes:
                mask_pred = (y_pred == cls)
                if np.any(mask_pred):
                    plt.scatter(x_data[mask_pred], np.full(mask_pred.sum(), class_to_num[cls]),
                                facecolors='none', #puste w środku
                                edgecolors=[class_to_color[cls]], #mają obwódki
                                linewidths=1.5, #grubość krawędzi
                                s=80, #wielkość punktu
                                alpha=0.9, #prawie nie są przezroczyste
                                label=f'Pred: {cls}')

            # Ustawiamy oś Y na wartości kategoryczne (czytelne)
            yticks = list(class_to_num.values())
            yticklabels = [num_to_class[i] for i in yticks]
            plt.yticks(yticks, yticklabels) #pozycje etykiet są na wysokości takiej, jak wcześniej zostały ustalone w class_to_num

            plt.title(f'Klasyfikacja: {target_name} vs {feature_name}', fontsize=14)
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel(target_name, fontsize=12)

            # Usuwamy duplikaty legendy (przy pierwszym wystąpieniu)
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys(), loc='best', fontsize='small')

            plt.grid(True, linestyle='--', alpha=0.7)

            plot_path = os.path.join('outputs', filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Wykres klasyfikacji zapisany: {plot_path}")

        except Exception as e:
            print(f"Błąd podczas generowania wykresu: {e}")

