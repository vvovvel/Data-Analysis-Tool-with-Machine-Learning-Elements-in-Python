from ml.ClassifierModel import ClassifierModel
from ml.ClusteringModel import ClusteringModel
from ml.RegressionModel import RegressionModel
from utils.context_manager import TimeLoggerContext
from utils.decorator import measure_time

@measure_time
def run_regression(df, id_col, target, features, test_size=0.2):
    with TimeLoggerContext("MODEL: LINEAR REGRESSION"):
        print("\n=== LINEAR REGRESSION ===")

        lin_model = RegressionModel(
            df=df,
            target_col=target,
            feature_cols=features,
            test_size=test_size
        )
        mse = lin_model.evaluate()
        print(f"MSE (test_size {test_size:.2f}): {mse:.2f}")

        print("\n Feature Impact Analysis")
        analysis_summary = lin_model.get_analysis_summary()

        print(f"Impact analysis for {target}:")
        for feature, data in analysis_summary["Regression Coefficients"].items():
            print(f"- {feature}: Impact: **{data['Impact']}**, Coefficient: {data['Value']:.2f}")

    return lin_model

@measure_time
def run_classification(df, id_col, target, features, n_neighbors=3, test_size=0.2):

    with TimeLoggerContext("MODEL: KNN CLASSIFICATION"):
        print("\n=== KNN CLASSIFICATION ===")

        knn_model = ClassifierModel(
            df=df,
            target_col=target,
            feature_cols=features,
            n_neighbors=n_neighbors,
            test_size=test_size
        )
        acc = knn_model.evaluate()
        print(f"Accuracy (n_neighbors={n_neighbors}): {acc:.2f}")

    return knn_model

@measure_time
def run_clustering(df, id_col, features, n_clusters=3):

    with TimeLoggerContext("MODEL: KMEANS CLUSTERING"):
        print("\n=== KMEANS CLUSTERING ===")

        kmeans_model = ClusteringModel(
            df=df,
            feature_cols=features,
            n_clusters=n_clusters,
        )
        score = kmeans_model.evaluate()
        print(f"Cluster quality score: {score}")

    return kmeans_model