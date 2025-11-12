import os
import pandas as pd
from sklearn.metrics import accuracy_score

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preprocessing import fill_na_with_value
from data.exceptions import InvalidDataError

from ml.LinearRegressionModel import LinearRegressionModel
from ml.KNNClasifierModel import KNNClassifierModel
from ml.KMeansClusteringModel import KMeansClusteringModel


def test_pipeline():
    DATA_PATH = os.path.join('data', 'Sleep_health_and_lifestyle_dataset.csv')

    REQUIRED_COLUMNS = [
        'Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
        'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
        'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
        'Sleep Disorder'
    ]

    POSITIVE_COLUMNS = [
        'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level',
        'Heart Rate', 'Daily Steps'
    ]

    SELECTED_FEATURES = ['Age']
    TARGET_COLUMN = 'Sleep Disorder'

    try:
        df = load_dataset(DATA_PATH)
        df = fill_na_with_value(df, ['Sleep Disorder'], 'None')
        validate_dataset(df, REQUIRED_COLUMNS, POSITIVE_COLUMNS)

        # --- ETAP 2: Model Regresji Liniowej ---
        print("\n=== Klasyfikacja ===")

        clasifier_model = KNNClassifierModel(
            df=df,
            target_col = TARGET_COLUMN,
            feature_cols = SELECTED_FEATURES,
            n_neighbors = 5,
            test_size = 0.2
        )

        accuracy_score = clasifier_model.evaluate()
        print(f'\nAccuracy: {accuracy_score}')

        # # 1. Model z podziałem 80/20 (test_size=0.2)
        # lin_model_20 = LinearRegressionModel(
        #     df=df,
        #     target_col=TARGET_COLUMN,
        #     feature_cols=SELECTED_FEATURES,
        #     test_size=0.2
        # )
        # mse_20 = lin_model_20.evaluate()
        # print(f"MSE (test_size 20%): {mse_20:.4f}")
        #
        # # 2. Model z podziałem 70/30 (test_size=0.3)
        # lin_model_30 = LinearRegressionModel(
        #     df=df,
        #     target_col=TARGET_COLUMN,
        #     feature_cols=SELECTED_FEATURES,
        #     test_size=0.3
        # )
        # mse_30 = lin_model_30.evaluate()
        # print(f"MSE (test_size 30%): {mse_30:.4f}")
        #
        # print("\n=== REGRESJA: Analiza Wpływu Zmiennych ===")
        #
        # # Analiza współczynników dla lepszego modelu (np. 80/20)
        # analysis_summary = lin_model_20.get_analysis_summary()
        #
        # print(f"Analiza wpływu na {TARGET_COLUMN}:")
        # for feature, data in analysis_summary["Współczynniki Regresji"].items():
        #     print(f"- {feature}: Wpływ: **{data['Wpływ']}**, Współczynnik: {data['Wartość']}")


    except InvalidDataError as e:
        print(f"Błąd w danych (InvalidDataError): {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas działania programu: {e}")


if __name__ == "__main__":
    test_pipeline()