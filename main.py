from data.loader import load_dataset
from data.preprocessing import fill_na_with_value
from data.validator import validate_dataset
from data.exceptions import InvalidDataError
from analysis.statistics import summary_stats

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
        'Heart Rate', 'Daily Steps'
    ]

    try:
        df = load_dataset()
        df = fill_na_with_value(df, ['Sleep Disorder'], 'None')
        validate_dataset(df)
        print("Pipeline działa – dane poprawne")
    except InvalidDataError as e:
        print(f"Błąd w danych: {e}")

    columns_to_analyze = ['Age', 'Sleep Duration', 'Daily Steps']
    try:
        stats = summary_stats(df, columns_to_analyze)
        print("Podstawowe statystyki:")
        for col, values in stats.items():
            print(f"\n{col}:")
            for k, v in values.items():
                print(f"  {k}: {v:.2f}")
    except InvalidDataError as e:
        print(f"Błąd analizy statystycznej: {e}")

if __name__ == "__main__":
    test_pipeline()

