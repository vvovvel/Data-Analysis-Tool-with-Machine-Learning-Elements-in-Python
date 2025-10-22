import pandas as pd
from data.exceptions import InvalidDataError

def summary_stats(df: pd.DataFrame, columns: list[str]) -> dict:
    stats = {}

    for col in columns:
        if col not in df.columns:
            raise InvalidDataError(f"Kolumna '{col}' nie istnieje w DataFrame.")

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Kolumna '{col}' nie zawiera wartości liczbowych.")

        stats[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }

    return stats
