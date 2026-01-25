import pandas as pd
from data.exceptions import InvalidDataError
from utils.context_manager import TimeLoggerContext
from utils.decorator import measure_time


def _summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:

    for col in columns:
        if col not in df.columns:
            raise InvalidDataError(f"Column '{col}' does not exist in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Column '{col}' is not numeric.")

    stats_df = df[columns].describe().T

    # Drop unnecessary columns to keep the summary concise
    stats_df = stats_df.drop(columns=['count', '25%', '50%', '75%'])

    return stats_df

# HELPER FUNCTION FOR THE NEXT METHOD
# pd.qcut() – Discretize variable into equal-sized buckets based on sample quantiles.
# Basic Syntax:
# pd.qcut(x, q, labels=None, duplicates='raise')
#
# Arguments:
# x        – Input array or Series to be cut.
# q        – Number of quantiles (e.g., 4 for quartiles) or list of quantiles [0, 0.25, 0.5, 0.75, 1.0].
# labels   – Labels for the resulting bins. If None, returns the interval.
# duplicates – 'raise' (default) -> error on non-unique bin edges,
#              'drop' -> drop non-unique bin edges to avoid errors.
#
# Example:
# df['AgeGroup'] = pd.qcut(df['Age'], q=4)  # 4 groups with equal number of observations.


def _grouped_mean_summary(
    df: pd.DataFrame,  # Input DataFrame
    group_col: str,    # Column to group by (e.g., Age to create age groups)
    target_col: str,   # Target column for calculation (e.g., Mean Sleep Duration)
    n_bins: int = 4     # Desired number of groups, must be within range <1,10>
) -> pd.DataFrame:

    if not isinstance(n_bins, int) or not (1 <= n_bins <= 10):
        raise InvalidDataError("n_bins must be an integer between 1 and 10.") # Manual input must be an int in <1,10>

    for col in [group_col, target_col]:
        if col not in df.columns:
            raise InvalidDataError(f"Missing column '{col}' from DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Column '{col}' is not numeric.")     # Both columns must be numeric

    df_copy = df.copy()

    # Divide into n_bins equal groups, each containing a similar number of observations
    df_copy['Group'] = pd.qcut(df_copy[group_col], q=n_bins, duplicates='drop')

    # Calculate mean for each of the groups
    result_series = (df_copy.groupby('Group')[target_col].mean()) # Returns a Series
    result_df = result_series.to_frame() # Convert to DataFrame
    result_df.rename(columns={target_col: f"Mean {target_col}"}, inplace=True) # Rename for clarity

    return result_df

def _corr_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:

    # If cols argument is not provided, calculate for all numeric columns
    if cols is None:
        cols = df.columns.tolist()

    # Validate columns
    for col in cols:
        if col not in df.columns:
            raise InvalidDataError(f"Missing column '{col}' from DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Column '{col}' is not numeric.")

    # Calculate the correlation matrix
    corr = df[cols].corr()

    return corr

@measure_time
def run_summary_stats(df, stats_columns):
    with TimeLoggerContext("DESCRIPTIVE STATISTICS"):

        print("\n=== STATISTICS: Basic Descriptive Statistics ===")

        summary_stats_result = _summary_stats(df, stats_columns)

        print(summary_stats_result.to_string(float_format='%.2f'))
        return summary_stats_result

@measure_time
def run_grouped_mean(df, group_col, target_col):
    with TimeLoggerContext("STATISTICS: Grouped Mean"):

        print(f"\n=== STATISTICS: Mean {target_col} by {group_col} ===")

        grouped_mean_result = _grouped_mean_summary(df, group_col, target_col)

        print(grouped_mean_result.to_string(float_format='%.2f'))
        return grouped_mean_result

@measure_time
def run_correlation_matrix(df, stats_columns):
    with TimeLoggerContext("STATISTICS: Correlation Matrix"):
        print("\n=== STATISTICS: Correlation Matrix ===")

        corr_matrix_result = _corr_matrix(df, stats_columns)

        print(corr_matrix_result.to_string(float_format='%.2f'))
        return corr_matrix_result