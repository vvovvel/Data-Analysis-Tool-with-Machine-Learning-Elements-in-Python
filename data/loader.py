import pandas as pd
import os #biblioteka do pracy z systemem plików
from data.exceptions import InvalidDataError

def load_dataset(file_path):
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