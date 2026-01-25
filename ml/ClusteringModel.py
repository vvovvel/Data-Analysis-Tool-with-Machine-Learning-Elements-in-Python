import numpy as np

from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The algorithm moves the cluster "centers" (centroids) until each data point is closest to its own centroid.

class ClusteringModel(BaseModel):

    def __init__(self, df=None, id_col=None ,feature_cols=None, n_clusters=3, random_state=42):
        self.id_col = id_col
        self.X_scaled = None
        self.feature_cols = None

        super().__init__(KMeans(n_clusters=n_clusters, random_state=random_state)) # Model imported from sklearn
        self.n_clusters = n_clusters # Number of clusters, i.e., groups

        if df is not None:
            preparer = DataPreparer(df, id_col)

            X_scaled = preparer.prepare_clustering(feature_cols=feature_cols)  # Data preparation for KMeans
            self.X_scaled = X_scaled
            self.feature_cols = X_scaled.columns.tolist()  # Saving feature column names

            self.train(X_scaled)  # Training the model


    def evaluate(self): # silhouette_score measures the quality of separation and consistency of clusters
        # Evaluates how similar a point is to its own cluster compared to other clusters. Values range from -1 to 1.
        try:
            X_data = self.X_scaled

            if X_data is None:
                raise ValueError("Model was not properly initialized with data.")

            labels = self.predict(X_data)

            # Minimum 2 unique clusters are required to calculate Silhouette Score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_data, labels)
                return {"Silhouette Score": round(score, 4)}
            else:
                return {"Info": f"Too few clusters ({self.n_clusters}) to calculate silhouette score"}

        except Exception as e:
            return {"Error": f"Cannot calculate silhouette score: {e}"}

    # Specific plot implementation

    def _draw_plot_content(self, plt):

        # 1. Data and feature validation (changed 'print' to 'raise ValueError' to be handled by BaseModel)
        # To draw the model, it must be previously initialized and trained
        if self.X_scaled is None:
            raise ValueError("Model must be initialized with data.")

        if self.feature_cols is None:
            raise ValueError("Missing feature names (feature_cols).")

        # Validation: exactly two features are required
        if len(self.feature_cols) != 2:
            raise ValueError(f"Visualization requires exactly 2 features. Found: {len(self.feature_cols)}.")

        # 2. Retrieving data and names from self attributes
        feature_name_1 = self.feature_cols[0]  # First feature
        feature_name_2 = self.feature_cols[1]  # Second feature

        # Data for plotting
        X_data = self.X_scaled.values
        labels = self.predict(self.X_scaled)  # Predicting labels
        n_unique_clusters = len(np.unique(
            labels))  # Usually n_unique_clusters == n_clusters, but can be less for small/asymmetric data

        # Clusters are required for plotting
        if n_unique_clusters < 1:
            raise ValueError("No clusters found for visualization.")

        # --- Plotting ---

        cmap = plt.get_cmap('Dark2', n_unique_clusters)  # Colormap for clusters

        # Scatter Plot (Data Points)
        for i in range(n_unique_clusters):  # Plotting points belonging to each cluster
            mask = (labels == i)  # Mask to get points only for the current cluster
            plt.scatter(X_data[mask, 0], X_data[mask, 1],  # Plotting each point in the cluster
                        label=f'Cluster {i}',
                        color=cmap(i),  # Selecting the appropriate color
                        alpha=0.6,
                        edgecolors='w',  # Outlines
                        linewidths=0.5)

        if self.model.cluster_centers_.any():  # Centroids, cluster centers
            centers_plot = self.model.cluster_centers_  # KMeans attribute: Numpy array of size n_clusters x 2 containing coordinates

            plt.scatter(centers_plot[:, 0], centers_plot[:, 1],  # Plotting centroids
                        marker='X', s=200,
                        color='black',
                        label='Centroids',
                        edgecolors='k',
                        linewidths=1.5)

        plt.title(f'Clustering Visualization: {feature_name_1} vs {feature_name_2}', fontsize=14)
        plt.xlabel(feature_name_1, fontsize=12)
        plt.ylabel(feature_name_2, fontsize=12)
        plt.legend(loc='best')  # Dynamic legend placement in the best possible location

    # Calling the base method which contains reusable code fragments for all models

    def plot(self, filename: str = "clustering_plot.png"):
        super().plot(filename=filename)