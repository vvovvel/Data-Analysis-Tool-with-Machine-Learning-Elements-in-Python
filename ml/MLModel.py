import pandas as pd
import numpy as np

class MLModel:
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