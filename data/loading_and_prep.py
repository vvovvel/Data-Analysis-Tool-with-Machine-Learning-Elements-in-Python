import os
import pandas as pd
from data.exceptions import InvalidDataError
from utils.context_manager import TimeLoggerContext


def _load_dataset(file_path):
    if not os.path.exists(file_path):
        raise InvalidDataError('File does not exist')

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise InvalidDataError("Dataset is empty")
    except pd.errors.ParserError:
        raise InvalidDataError("CSV format error")
    except PermissionError:
        raise InvalidDataError("Permission denied to read file")
    except Exception as exc:
        raise InvalidDataError(f"Unexpected error: {exc}")

    return df


def _fill_na_with_value(df: pd.DataFrame, columns: list, fill_value):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


def _validate_dataset(df: pd.DataFrame, required_columns: list, positive_columns: list):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidDataError(f"Missing columns in dataset: {missing_columns}")

    for col in positive_columns:
        if (df[col] <= 0).any():
            raise InvalidDataError(f"Invalid (negative or zero) values in column: {col}")

    for col in required_columns:
        if df[col].isnull().any():
            raise InvalidDataError(f"Missing values in column: {col}")

    return True

def perform_loading_and_prep(data_path, required_cols, fill_na_cols, fill_value, positive_cols):

    with TimeLoggerContext("LOADING AND PREPROCESSING"):
        df = _load_dataset(data_path)
        df = _fill_na_with_value(df, fill_na_cols, fill_value)
        _validate_dataset(df, required_cols, positive_cols)

    return df
