import pandas as pd
from data.exceptions import InvalidDataError

def validate_dataset(df: pd.DataFrame, required_columns: list, positive_columns: list):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidDataError(f"Brakujące kolumny w dataset: {missing_columns}")

    for col in positive_columns:
        if (df[col] <= 0).any():
            raise InvalidDataError(f"Niepoprawne (ujemne lub zerowe) wartości w kolumnie: {col}")

    for col in required_columns:
        if df[col].isnull().any():
            raise InvalidDataError(f"Brakujące wartości w kolumnie: {col}")

    return True