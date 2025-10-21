import pandas as pd
from data.exceptions import InvalidDataError

# Lista kolumn, które powinny być w dataset
REQUIRED_COLUMNS = [
    'Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
    'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
    'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
    'Sleep Disorder'
]

# Kolumny, które muszą mieć tylko wartości dodatnie
POSITIVE_COLUMNS = [
    'Age', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level',
    'Heart Rate', 'Daily Steps'
]

def validate_dataset(df: pd.DataFrame):

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise InvalidDataError(f"Brakujące kolumny w dataset: {missing_columns}")

    for col in POSITIVE_COLUMNS:
        if (df[col] <= 0).any():
            raise InvalidDataError(f"Niepoprawne (ujemne lub zerowe) wartości w kolumnie: {col}")

    for col in REQUIRED_COLUMNS:
        if df[col].isnull().any():
            raise InvalidDataError(f"Brakujące wartości w kolumnie: {col}")

    return True
