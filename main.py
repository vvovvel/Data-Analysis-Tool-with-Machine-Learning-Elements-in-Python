import os

from data.loading_and_prep import perform_loading_and_prep
from data.exceptions import InvalidDataError

from analysis.statistics import run_summary_stats, run_grouped_mean, run_correlation_matrix
from analysis.plots import plot_corr_matrix, plot_data

from ml.ModelRunner import run_regression, run_classification, run_clustering



# def test_pipeline_sleep():
#
#     DATA_PATH = os.path.join('data', 'Sleep_health_and_lifestyle_dataset.csv')
#
#     REQUIRED_COLUMNS = [
#         'Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
#         'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
#         'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
#         'Sleep Disorder'
#     ]
#     POSITIVE_COLUMNS = [
#         'Age', 'Sleep Duration', 'Quality of Sleep',
#         'Physical Activity Level', 'Stress Level',
#         'Heart Rate', 'Daily Steps'
#     ]
#
#     ID_COL = 'Person ID'
#
#     FILL_NA_COLS = ['Sleep Disorder']
#
#     FILL_NA_VALUE = 'None'
#
#     STATS_COLUMNS = ['Age', 'Sleep Duration', 'Stress Level', 'Daily Steps', 'Heart Rate']
#
#     GROUPED_MEAN_COL = 'Quality of Sleep'
#     GROUPED_MEAN_TARGET = 'Sleep Duration'
#
#     SCATTER_X = 'Age'
#     SCATTER_Y = 'Stress Level'  # np. 'Stress Level'
#
#     HISTOGRAM_X = 'Sleep Duration'
#
#     BOXPLOT_X = 'BMI Category'
#     BOXPLOT_Y = 'Sleep Duration'
#
#     REGRESSION_FEATURES = ['Age']
#     REGRESSION_TARGET = 'Stress Level'
#
#     CLASSIFICATION_FEATURES = ['Physical Activity Level']
#     CLASSIFICATION_TARGET = 'Sleep Disorder'
#
#     CLUSTERING_FEATURES = ['Heart Rate', 'Daily Steps']
#
#     try:
#
#         df = perform_loading_and_prep(DATA_PATH, REQUIRED_COLUMNS, FILL_NA_COLS, FILL_NA_VALUE, POSITIVE_COLUMNS)
#
#         summary_stats = run_summary_stats(df, STATS_COLUMNS)
#         grouped_mean = run_grouped_mean(df, GROUPED_MEAN_COL, GROUPED_MEAN_TARGET)
#         corr_matrix = run_correlation_matrix(df, STATS_COLUMNS)
#
#         plot_corr_matrix(corr_matrix)
#         plot_data(df, 'scatter', SCATTER_X, SCATTER_Y)
#         plot_data(df, 'histogram', HISTOGRAM_X)
#         plot_data(df, 'boxplot', BOXPLOT_X, BOXPLOT_Y)
#
#         lin_model = run_regression(df, ID_COL, REGRESSION_TARGET, REGRESSION_FEATURES)
#         lin_model.plot()
#
#         knn_model = run_classification(df, ID_COL, CLASSIFICATION_FEATURES)
#         knn_model.plot()
#
#         kmeans_model = run_clustering(df, ID_COL, CLUSTERING_FEATURES)
#         kmeans_model.plot()
#
#
#     except InvalidDataError as e:
#         print(f"Błąd w danych (InvalidDataError): {e}")
#     except Exception as e:
#         print(f"Wystąpił nieoczekiwany błąd podczas działania programu: {e}")


def test_pipeline_exams():

    DATA_PATH = os.path.join('data', 'student_exam_scores.csv')

    REQUIRED_COLUMNS = ['student_id', 'hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score', 'exam_grade']

    POSITIVE_COLUMNS = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score']

    ID_COL = 'student_id'

    FILL_NA_COLS = []

    FILL_NA_VALUE = 0

    STATS_COLUMNS = ['hours_studied', 'sleep_hours', 'previous_scores', 'exam_score']

    GROUPED_MEAN_COL = 'hours_studied'
    GROUPED_MEAN_TARGET = 'exam_score'

    SCATTER_X = 'hours_studied'
    SCATTER_Y = 'exam_score'

    HISTOGRAM_X = 'exam_score'

    BOXPLOT_X = 'exam_grade'
    BOXPLOT_Y = 'hours_studied'

    REGRESSION_FEATURES = ['hours_studied']
    REGRESSION_TARGET = 'exam_score'

    CLASSIFICATION_FEATURES = ['hours_studied']
    CLASSIFICATION_TARGET = 'exam_grade'

    CLUSTERING_FEATURES = ['hours_studied', 'attendance_percent']

    try:

        df = perform_loading_and_prep(DATA_PATH, REQUIRED_COLUMNS, FILL_NA_COLS, FILL_NA_VALUE, POSITIVE_COLUMNS)

        summary_stats = run_summary_stats(df, STATS_COLUMNS)
        grouped_mean = run_grouped_mean(df, GROUPED_MEAN_COL, GROUPED_MEAN_TARGET)
        corr_matrix = run_correlation_matrix(df, STATS_COLUMNS)

        plot_corr_matrix(corr_matrix)
        plot_data(df, 'scatter', SCATTER_X, SCATTER_Y)
        plot_data(df, 'histogram', HISTOGRAM_X)
        plot_data(df, 'boxplot', BOXPLOT_X, BOXPLOT_Y)

        lin_model = run_regression(df, ID_COL, REGRESSION_TARGET, REGRESSION_FEATURES)
        lin_model.plot("linear_students")

        knn_model = run_classification(df, ID_COL, CLASSIFICATION_TARGET, CLASSIFICATION_FEATURES)
        knn_model.plot("knn_students")

        kmeans_model = run_clustering(df, ID_COL, CLUSTERING_FEATURES, 5)
        kmeans_model.plot("cluster_students")



    except InvalidDataError as e:

        print(f"Data error (InvalidDataError): {e}")

    except Exception as e:

        print(f"An unexpected error occurred during program execution: {e}")


if __name__ == "__main__":
    test_pipeline_exams()
