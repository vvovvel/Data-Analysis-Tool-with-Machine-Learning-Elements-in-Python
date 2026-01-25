import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreparer:
    def __init__(self, df: pd.DataFrame, id_col):
        self.df = df.copy()
        self.id_col = id_col

    def _validate_columns(self, cols):
        # Checks if all columns exist in the DataFrame
        missing = []
        for c in cols:
            if c not in self.df.columns:
                missing.append(c)
        if len(missing) > 0:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

    def _choose_features(self, target_col=None, feature_cols=None):
        if feature_cols is not None:
            # Validation
            self._validate_columns(feature_cols)
            cols = []
            for c in feature_cols:
                if c != self.id_col:
                    cols.append(c)
            # Remove target if it was mistakenly included in feature_cols
            if target_col is not None and target_col in cols:
                cols.remove(target_col)
            if len(cols) == 0:
                raise ValueError("No feature columns remaining after removing ID/target.")
            X = self.df[cols]
        else:
            # Take all columns except ID and target (if target is provided)
            drop_cols = []
            if self.id_col in self.df.columns:
                drop_cols.append(self.id_col)
            if target_col is not None and target_col in self.df.columns:
                drop_cols.append(target_col)
            X = self.df.drop(columns=drop_cols)
            if X.shape[1] == 0:
                raise ValueError("No feature columns remaining after dropping ID and target.")
        return X

    def prepare_data_regression(self, target_col, feature_cols=None, scale=False,
                                test_size=0.2, random_state=42):
        # Returns DataFrames (scales if requested)
        X = self._choose_features(target_col=target_col, feature_cols=feature_cols)
        y = self.df[target_col]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Convert back to DataFrame with the same columns and indices
            X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        return X, y, X_train, X_test, y_train, y_test

    def prepare_classification(self, target_col, feature_cols=None, test_size=0.2, random_state=81):
        # Returns DataFrames
        return self.prepare_data_regression(
            target_col=target_col,
            feature_cols=feature_cols,
            scale=True, # The only difference from regression is mandatory scaling
            test_size=test_size,
            random_state=random_state
        )

    def prepare_clustering(self, feature_cols=None):
        # Choose raw features (skipping id_col)
        X = self._choose_features(target_col=None, feature_cols=feature_cols)

        # Categorical encoding and scaling the entire dataset
        X = pd.get_dummies(X, drop_first=True)
        if X.shape[1] == 0:
            raise ValueError("No feature columns for clustering after dropping id_col.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index) # Convert back to DataFrame
        return X_scaled