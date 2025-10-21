import pandas as pd

def fill_na_with_value(df: pd.DataFrame, columns: list, fill_value):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df
