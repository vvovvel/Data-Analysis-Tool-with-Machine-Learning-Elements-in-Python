import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreparer:
    def __init__(self, df: pd.DataFrame, id_col='Person ID'):
        self.df = df.copy()
        self.id_col = id_col

    def _validate_columns(self, cols):
        #sprawdza czy wszystkie kolumny isntnieją
        missing = []
        for c in cols:
            if c not in self.df.columns:
                missing.append(c)
        if len(missing) > 0:
            raise ValueError(f"Brakuje kolumn w df: {missing}")

    def _choose_features(self, target_col=None, feature_cols=None):
        if feature_cols is not None:
            # walidacja
            self._validate_columns(feature_cols)
            cols = []
            for c in feature_cols:
                if c != self.id_col:
                    cols.append(c)
            # usuń target, jeśli ktoś go podał w feature_cols przez pomyłkę
            if target_col is not None and target_col in cols:
                cols.remove(target_col)
            if len(cols) == 0:
                raise ValueError("Po usunięciu id/targetu nie pozostały kolumny cech.")
            X = self.df[cols]
        else:
            # bierzemy wszystkie kolumny poza id i target (jeśli target podano)
            drop_cols = []
            if self.id_col in self.df.columns:
                drop_cols.append(self.id_col)
            if target_col is not None and target_col in self.df.columns:
                drop_cols.append(target_col)
            X = self.df.drop(columns=drop_cols)
            if X.shape[1] == 0:
                raise ValueError("Brak kolumn cech po odrzuceniu id i targetu.")
        return X

    def prepare_data_regression(self, target_col, feature_cols=None, scale=False,
                                test_size=0.2, random_state=42):
        X = self._choose_features(target_col=target_col, feature_cols=feature_cols)
        y = self.df[target_col]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        return X, y, X_train, X_test, y_train, y_test

    def prepare_classification(self, target_col, feature_cols=None, test_size=0.2, random_state=81):

        return self.prepare_data_regression(
            target_col=target_col,
            feature_cols=feature_cols,
            scale=True, #jedyne co się zmienia względem LinearRegression to obowiązkowe skalowanie
            test_size=test_size,
            random_state=random_state
        )

    def prepare_clustering(self, feature_cols=None):
        # wybór surowych cech (pomijamy id_col)
        X = self._choose_features(target_col=None, feature_cols=feature_cols)

        # kodowanie kategorii i skalowanie całego zbioru
        X = pd.get_dummies(X, drop_first=True)
        if X.shape[1] == 0:
            raise ValueError("Brak kolumn cech do klasteryzacji po odrzuceniu id_col.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
