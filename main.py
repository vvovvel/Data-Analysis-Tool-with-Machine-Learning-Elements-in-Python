
import os #biblioteka do pracy z systemem plików
from data.loader import load_dataset
from data.validator import validate_dataset
from data.preprocessing import fill_na_with_value
from data.exceptions import InvalidDataError
from analysis.statistics import summary_stats
from analysis.statistics import grouped_mean_summary_auto

def test_pipeline():

    DATA_PATH = os.path.join('data', 'Sleep_health_and_lifestyle_dataset.csv')
    # oznacza to ścieżkę, ale zapisaną zarówno w Linux/iOS/Windows która łączy folder data z plikiem Sleep_health..
    # to gwarantuje, że ścieżka do pliku jest poprawna względem katalogu głównego projektu
    # DATA_PATH z wielkich liter zgodnie z PEP 8, bo jest const

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
        'Heart Rate', 'Daily Steps']

    try:
        df = load_dataset(DATA_PATH)  # wczytanie CSV
        df = fill_na_with_value(df, ['Sleep Disorder'], 'None')  # brakujące Sleep Disorder -> 'None'
        validate_dataset(df, REQUIRED_COLUMNS, POSITIVE_COLUMNS)  # walidacja

        # przykład użycia grouped_mean_summary_auto
        # podział osób na 4 grupy według wieku i średnia długość snu
        #automatyczne wypisywanie
        grouped_mean_summary_auto(df, group_col='Physical Activity Level', target_col='Quality of Sleep', n_bins=4)

    except InvalidDataError as e:
        print(f"Błąd w danych: {e}")


if __name__ == "__main__":
    test_pipeline()

