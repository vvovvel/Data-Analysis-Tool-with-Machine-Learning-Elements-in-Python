import pandas as pd
import os #biblioteka do pracy z systemem plików

from data.exceptions import InvalidDataError

DATA_PATH = os.path.join('data', 'Sleep_health_and_lifestyle_dataset.csv')
#oznacza to ścieżkę, ale zapisaną zarówno w Linux/iOS/Windows która łączy folder data z plikiem Sleep_health..
#to gwarantuje, że ścieżka do pliku jest poprawna względem katalogu głównego projektu
#DATA_PATH z wielkich liter zgodnie z PEP 8, bo jest const

def load_dataset(file_path = DATA_PATH):
    if not os.path.exists(file_path):
        raise InvalidDataError('File does not exist')

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise InvalidDataError("Dataset is empty")
    except pd.errors.ParserError:  #błędny csv np zamiast ; jest , albo rozne liczby kolumn
        raise InvalidDataError("Problem z formatem CSV")
    except PermissionError:
        raise InvalidDataError("Brak uprawnień do odczytu pliku")
    except Exception as exc: #wszystkie inne wyjątki
        raise InvalidDataError(f"Nieoczekiwany błąd: {exc}")

    return df