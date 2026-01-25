import pandas as pd
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel(ABC):
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
            print(f"[ERROR] Training error: {e}")
            raise e

    def predict(self, X_test):
        try:
            return self.model.predict(X_test)
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            raise e

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def _draw_plot_content(self, plt): #specifiic way to draw every plot
        pass

    def plot(self, filename: str = "model_plot.png"): #common logic for every plot
        try:
            os.makedirs('outputs', exist_ok=True) #checking outputs directory
            plt.figure(figsize=(10, 6)) #common size

            #specific metod for every plot
            self._draw_plot_content(plt)

            #common style elements
            plt.grid(True, linestyle='--', alpha=0.7)

            #zapis i czyszczenie
            plot_path = os.path.join('outputs', filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Model plot saved: {plot_path}")

        except ValueError as e:
            # catching validation errors
            print(f"Plot data validation error: {e}")
            plt.close()  #closing figure
            raise e
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close()
            raise e