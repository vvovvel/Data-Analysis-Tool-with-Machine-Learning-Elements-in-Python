import pandas as pd
import pytest
from data.validator import validate_dataset
from data.exceptions import InvalidDataError

def make_valid_df():
    return pd.DataFrame({
        'Person ID': [1, 2],
        'Gender': ['M', 'F'],
        'Age': [25, 30],
        'Occupation': ['A', 'B'],
        'Sleep Duration': [7.0, 8.0],
        'Quality of Sleep': [3, 4],
        'Physical Activity Level': [2, 3],
        'Stress Level': [1, 2],
        'BMI Category': ['Normal', 'Overweight'],
        'Blood Pressure': ['120/80', '130/85'],
        'Heart Rate': [70, 75],
        'Daily Steps': [5000, 6000],
        'Sleep Disorder': ['None', 'Insomnia']
    })

REQUIRED_COLUMNS = [
    'Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
    'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
    'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Sleep Disorder'
]

POSITIVE_COLUMNS = [
    'Age', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps'
]

def test_validator_accepts_valid_df():
    df = make_valid_df()
    assert validate_dataset(df, REQUIRED_COLUMNS, POSITIVE_COLUMNS) is True  #brak błędów

def test_validator_detects_various_errors():
    df = make_valid_df()

    df.loc[0, 'Age'] = -5
    with pytest.raises(InvalidDataError):
        validate_dataset(df, REQUIRED_COLUMNS, POSITIVE_COLUMNS)   #błąd z powodu ujemnej wartości

    df.loc[0, 'Age'] = 25

    df.loc[1, 'Sleep Duration'] = None
    with pytest.raises(InvalidDataError):
        validate_dataset(df, REQUIRED_COLUMNS, POSITIVE_COLUMNS) #błąd z powodu braku wartości

    df_missing = df.drop(columns=['Gender'])
    with pytest.raises(InvalidDataError):
        validate_dataset(df_missing, REQUIRED_COLUMNS, POSITIVE_COLUMNS) #błąd z powodu braku kolumny