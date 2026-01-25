import numpy as np

from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN stands for K-Nearest Neighbors -> the model finds the K closest neighbors and assigns the class most frequently represented among them.
# In this model, target_col must contain categorical variables, e.g., occupation, BMI category, or sleep disorder.
# The model is initialized by providing the DataFrame, target_col, and feature_cols (which defaults to all columns except the target and ID).
# We can then access the accuracy, which is the ratio of correct predictions to the total number of samples.

class ClassifierModel(BaseModel):

    def __init__(self, df=None, id_col=None, target_col=None, feature_cols=None, n_neighbors=5, test_size=0.2):
        self.id_col = id_col
        self.X_test = None
        self.y_test = None
        self.feature_cols = None
        self.target_col = None

        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors)) # Model imported from sklearn
        self.n_neighbors = n_neighbors
        self.target_col = target_col

        if df is not None and target_col is not None:
            preparer = DataPreparer(df, id_col)

            # Using preparer.prepare_classification (which scales the data)
            X, y, X_train, self.X_test, y_train, self.y_test = preparer.prepare_classification(
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_size
            )

            # Training the model
            self.train(X_train, y_train)
            self.feature_cols = X_train.columns.tolist()

    def evaluate(self):
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Model was not properly initialized with data.")

            y_pred = self.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return accuracy

        except Exception as e:
            return {"Error": f"Cannot calculate Accuracy: {e}"}

    #specific plot implementation

    def _draw_plot_content(self, plt):

        if self.X_test is None or self.y_test is None:
            raise ValueError("Model must be initialized.")

        if self.feature_cols is None:
            raise ValueError("Missing feature names (feature_cols).")

        if len(self.feature_cols) != 1:
            raise ValueError(f"Visualization requires exactly 1 feature. Found: {len(self.feature_cols)}.")

        # Extracting names
        feature_name = self.feature_cols[0]
        target_name = self.target_col

        # Preparing axis data
        # Selecting the ONLY column from X_test and converting to a 1D vector
        x_data = self.X_test[feature_name].values.flatten()
        y_test_data = np.array(self.y_test).flatten()

        # Calculating predictions
        y_pred = np.array(self.predict(self.X_test)).flatten()

        # Class mapping (e.g., ['low','medium','high'] -> [0,1,2])
        classes = np.unique(
            np.concatenate([y_test_data, y_pred]))  # Combine test and predicted data into a unique set
        class_to_num = {cls: i for i, cls in enumerate(
            classes)}  # Dictionary -> assigns a natural number to each unique value in classes
        num_to_class = {i: cls for cls, i in
                        class_to_num.items()}  # Inverse mapping: each number maps back to the value (e.g., string)

        # Plotting

        # Assigning colors to classes
        n_unique_classes = len(classes)  # Number of unique classes
        cmap = plt.get_cmap('Dark2', n_unique_classes)  # Get exactly as many colors as needed
        class_to_color = {cls: cmap(i) for i, cls in enumerate(classes)}  # Assign color to each class

        # Plot true labels (filled)
        for cls in classes:
            mask = (y_test_data == cls)  # Mask creating a boolean array for each class... [low, high, low, medium] -> [True, False, True, False] for 'low'.
            # Basically -> True if it's this class, False otherwise BOOLEAN INDEXING
            plt.scatter(x_data[mask], np.full(mask.sum(), class_to_num[cls]),
                        # Select only values for points belonging to this class (arguments).
                        # Create y-values (array matching the point count), all set to the height defined by class_to_num (categorical values mapped to integers).
                        color=class_to_color[cls],  # filled
                        alpha=0.6,  # semi-transparent
                        label=f'True: {cls}')

        # Plot predictions
        for cls in classes:
            mask_pred = (y_pred == cls)
            if np.any(mask_pred):
                plt.scatter(x_data[mask_pred], np.full(mask_pred.sum(), class_to_num[cls]),
                            facecolors='none',  # hollow
                            edgecolors=[class_to_color[cls]],  # with outlines
                            linewidths=1.5,  # edge width
                            s=80,  # point size
                            alpha=0.9,  # almost opaque
                            label=f'Pred: {cls}')

        # Set Y-axis to categorical values (readable)
        yticks = list(class_to_num.values())
        yticklabels = [num_to_class[i] for i in yticks]
        plt.yticks(yticks, yticklabels)  # Label positions match the heights previously defined in class_to_num

        plt.title(f'Classification Visualization: {target_name} vs {feature_name}', fontsize=14)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel(target_name, fontsize=12)

        # Remove duplicate legend entries (keep first occurrence)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), loc='best', fontsize='small')

    # Call the base method containing common code logic for all models

    def plot(self, filename: str = "classifier_plot.png"):
        super().plot(filename=filename)

