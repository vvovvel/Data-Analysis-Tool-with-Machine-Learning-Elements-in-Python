import numpy as np
from ml.BaseModel import BaseModel
from ml.DataPreparer import DataPreparer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils.context_manager import TimeLoggerContext

# df -> dataframe, our default is SleepHealth
# target_col -> what we want to predict, must be numeric
# feature_cols -> columns used for prediction, can be any but cannot include target_col 
# (even if included, it will be removed in DataPreparer)
# optionally we can set test_size (train/test split ratio)
# once initialized, it creates the whole class, including data splitting
# we can extract MSE or coefficients to find out if a variable has a positive or negative impact

class RegressionModel(BaseModel):

    # Constructor
    def __init__(self, df=None, id_col=None, target_col=None, feature_cols=None, test_size=0.2):
        # Save only specific variables needed for other methods like evaluate, plot
        self.id_col = id_col
        self.X_test = None
        self.y_test = None
        self.feature_cols = None
        self.target_col = None
        self.coefficients = None
        self.test_size = None

        # Call base class constructor
        super().__init__(LinearRegression()) # Refers to BaseModel and calls its constructor with LinearRegression model

        if df is not None and target_col is not None: # Basic data validation
            preparer = DataPreparer(df, id_col) # Create DataPreparer object

            X, y, X_train, self.X_test, y_train, self.y_test = preparer.prepare_data_regression(
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_size
            ) 

            # Train the model
            self.train(X_train, y_train)

            # Save analysis results after training
            self.target_col = target_col
            self.feature_cols = X_train.columns.tolist()
            self.coefficients = self.model.coef_

    def evaluate(self):
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Model was not properly initialized with data.")

            y_pred = self.predict(self.X_test) # predict is in BaseModel but calls the internal LinearRegression model
            mse = mean_squared_error(self.y_test, y_pred)
            return mse
        except Exception as e:
            return {"Error": f"Cannot calculate MSE: {e}"}

    def get_analysis_summary(self):
        if self.coefficients is None:
            return "Model has not been trained yet."

        summary = {"Regression Coefficients": {}}

        for name, coef in zip(self.feature_cols, self.coefficients): # zip function pairs elements from two lists by index

            if coef > 0:
                impact = "Positive"
            elif coef < 0:
                impact = "Negative"
            else:
                impact = "No impact"

            summary["Regression Coefficients"][name] = {
                "Impact": impact,
                "Value": round(coef, 4)
            }

        return summary

    # Specific plot implementation
    def _draw_plot_content(self, plt):

        if self.X_test is None or self.y_test is None:
            raise ValueError("Model must be initialized.")

        if self.feature_cols is None:
            raise ValueError("Missing feature names (feature_cols).")

        if len(self.feature_cols) != 1:
            raise ValueError(f"Visualization requires exactly 1 feature. Found: {len(self.feature_cols)}.")

        # 2. Extracting data and names from self attributes
        feature_name = self.feature_cols[0]  # ONLY feature
        target_name = self.target_col  # Target Name

        # 3. Preparing axis data
        x_data = self.X_test[feature_name].values.flatten()
        y_test_data = self.y_test.values.flatten()

        # 4. Calculating regression line predictions
        y_pred = self.predict(self.X_test).flatten()

        # --- Plotting ---
        # Scatter Plot (Data Points)
        plt.scatter(x_data, y_test_data, color='#3498db', alpha=0.6, label='Test Data')

        # Regression Line (Prediction)
        sort_idx = x_data.argsort()
        plt.plot(x_data[sort_idx], y_pred[sort_idx], color='#e74c3c', linewidth=3, label='Regression Line')

        plt.title(f'Linear Regression Visualization: {target_name} vs {feature_name}', fontsize=14)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        plt.legend()

    # Call the base method containing common code logic for all models
    def plot(self, filename: str = "regression_plot.png"):
        super().plot(filename=filename)