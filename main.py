import os

from data.loader import load_dataset
from data.validator import validate_dataset
from data.preprocessing import fill_na_with_value
from data.exceptions import InvalidDataError

from analysis import statistics as stat

from ml.LinearRegressionModel import LinearRegressionModel
from ml.KNNClasifierModel import KNNClassifierModel
from ml.KMeansClusteringModel import KMeansClusteringModel


from utils.context_managers import TimeLoggerContext


def _perform_loading_and_prep(data_path, required_cols, positive_cols):

    with TimeLoggerContext("ŁADOWANIE i PREPROCESSING"):
        df = load_dataset(data_path)
        df = fill_na_with_value(df, ['Sleep Disorder'], 'None')
        validate_dataset(df, required_cols, positive_cols)

    return df


def _run_summary_stats(df, stats_columns):
    with TimeLoggerContext("STATYSTYKI OPISOWE"):

        print("\n=== STATYSTYKI: Podstawowe Statystyki Opisowe ===")

        summary_stats_result = stat.summary_stats(df, stats_columns)

        print(summary_stats_result.to_string(float_format='%.2f'))
        return summary_stats_result


def _run_grouped_mean(df, group_col, target_col):
    with TimeLoggerContext("STATYSTYKI: Średnia Grupowa"):

        print(f"\n=== STATYSTYKI: Średnia {target_col} wg {group_col} ===")

        grouped_mean_result = stat.grouped_mean_summary_auto(df, group_col, target_col)

        print(grouped_mean_result.to_string(float_format='%.2f'))
        return grouped_mean_result


def _run_correlation_matrix(df, stats_columns):
    with TimeLoggerContext("STATYSTYKI: Macierz Korelacji"):
        print("\n=== STATYSTYKI: Macierz Korelacji ===")

        corr_matrix_result = stat.corr_matrix(df, stats_columns)

        print(corr_matrix_result.to_string(float_format='%.2f'))
        return corr_matrix_result


def _run_regression(df, target, features):

    with TimeLoggerContext("MODEL: REGRESJA LINIOWA"):
        print("\n=== REGRESJA LINIOWA ===")

        lin_model_20 = LinearRegressionModel(
            df=df,
            target_col=target,
            feature_cols=features,
            test_size=0.2
        )
        mse_20 = lin_model_20.evaluate()
        print(f"MSE (test_size 20%): {mse_20:.4f}")

        print("\n Analiza Wpływu Zmiennych")
        analysis_summary = lin_model_20.get_analysis_summary()

        print(f"Analiza wpływu na {target}:")
        for feature, data in analysis_summary["Współczynniki Regresji"].items():
            print(f"- {feature}: Wpływ: **{data['Wpływ']}**, Współczynnik: {data['Wartość']}")

    return lin_model_20


def _run_classification(df, target, features):

    with TimeLoggerContext("MODEL: KLASYFIKACJA KNN"):
        print("\n=== KLASYFIKACJA KNN ===")

        knn_model_3 = KNNClassifierModel(
            df=df,
            target_col=target,
            feature_cols=features,
            n_neighbors=3,
            test_size=0.2
        )
        acc_3 = knn_model_3.evaluate()
        print(f"Dokładność (n_neighbors=3): {acc_3:.4f}")

    return knn_model_3


def _run_clustering(df, features):

    with TimeLoggerContext("MODEL: KLASTERYZACJA KMeans"):
        print("\n=== KLASTERYZACJA KMeans ===")

        kmeans_model_3 = KMeansClusteringModel(
            df=df,
            feature_cols=features,
            n_clusters=3
        )
        score = kmeans_model_3.evaluate()
        print(f"Ocena jakości klastrów: {score}")

    return kmeans_model_3


#pipEline

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

    STATS_COLUMNS = ['Age', 'Sleep Duration', 'Stress Level', 'Daily Steps', 'Heart Rate']

    GROUPED_MEAN_COL = 'Age'

    GROUPED_MEAN_TARGET = 'Daily Steps'

    REGRESSION_FEATURES = ['Age', 'Physical Activity Level', 'Stress Level']
    REGRESSION_TARGET = 'Sleep Duration'

    CLASSIFICATION_FEATURES = ['Age', 'Quality of Sleep', 'Stress Level', 'BMI Category', 'Daily Steps']
    CLASSIFICATION_TARGET = 'Sleep Disorder'

    CLUSTERING_FEATURES = ['Age', 'Stress Level', 'Daily Steps', 'Heart Rate']

    try:

        df = _perform_loading_and_prep(DATA_PATH, REQUIRED_COLUMNS, POSITIVE_COLUMNS)

        summary_stats = _run_summary_stats(df, STATS_COLUMNS)

        grouped_mean = _run_grouped_mean(df, GROUPED_MEAN_COL, GROUPED_MEAN_TARGET)

        corr_matrix = _run_correlation_matrix(df, STATS_COLUMNS)

        lin_model = _run_regression(df, REGRESSION_TARGET, REGRESSION_FEATURES)

        knn_model = _run_classification(df, CLASSIFICATION_TARGET, CLASSIFICATION_FEATURES)

        kmeans_model = _run_clustering(df, CLUSTERING_FEATURES)


    except InvalidDataError as e:
        print(f"Błąd w danych (InvalidDataError): {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas działania programu: {e}")


if __name__ == "__main__":
    test_pipeline()