from data.loader import load_dataset
from data.preprocessing import fill_na_with_value
from data.validator import validate_dataset
from data.exceptions import InvalidDataError

def test_pipeline():
    try:
        df = load_dataset()
        df = fill_na_with_value(df, ['Sleep Disorder'], 'None')
        validate_dataset(df)
        print("Pipeline działa – dane poprawne")
    except InvalidDataError as e:
        print(f"Błąd w danych: {e}")

if __name__ == "__main__":
    test_pipeline()

