import pandas as pd
from data.exceptions import InvalidDataError

def summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:

    for col in columns:
        if col not in df.columns:
            raise InvalidDataError(f"Kolumna '{col}' nie istnieje w DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Kolumna '{col}' nie zawiera wartości liczbowych.")

    stats_df = df[columns].describe().T

    stats_df = stats_df.drop(columns=['count', '25%', '50%', '75%'])

    return stats_df

#FUNKCJA UŻYTECZNA DO KOLEJNEJ METODY
# pd.qcut() – dzieli kolumnę numeryczną na "kwantyle" (grupy o podobnej liczbie obserwacji)
# Składnia podstawowa:
# pd.qcut(x, q, labels=None, duplicates='raise')
#
# Argumenty:
# x        – kolumna/Series do podziału
# q        – liczba kwantyli (np. 4 → podział na ćwiartki) lub lista wartości kwantyli [0, 0.25, 0.5, 0.75, 1.0]
# labels   – etykiety dla grup; jeśli None → pandas tworzy Interval
# duplicates – 'raise' (domyślnie) → błąd przy powtarzających się granicach,
#              'drop' → usuwa duplikaty, żeby uniknąć błędu
#
# Przykład:
# df['AgeGroup'] = pd.qcut(df['Age'], q=4)  # 4 grupy z równą liczbą osób


def grouped_mean_summary_auto(
    df: pd.DataFrame,  #wkładamy data frame
    group_col: str,    #ze względu na co chcemy grupować np ze względu na wiek, grupy wiekowe 25-30, 30-40 itd
    target_col: str,    #co chcemy mieć w tym grupowaniu np długość snu dla każdej grupy wiekowej, ilość kroków itd
    n_bins: int = 4     #wpisujemy sobie ilość grup, którą chcemy, ale musi być z przedziału <1,10>
) -> pd.DataFrame:

    if not isinstance(n_bins, int) or not (1 <= n_bins <= 10):
        raise InvalidDataError("n_bins musi być liczbą całkowitą od 1 do 10.") #liczba grup będzie wpisywana ręcznie ale musi być intem z <1,10>

    for col in [group_col, target_col]:
        if col not in df.columns:
            raise InvalidDataError(f"Brak kolumny '{col}' w DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Kolumna '{col}' nie zawiera wartości liczbowych.")     #obie kolumny oczywiście muszą być w df

    df_copy = df.copy()

    df_copy['Group'] = pd.qcut(df_copy[group_col], q=n_bins, duplicates='drop')  #dzieli na cztery równe grupy, każda zawiera tyle samo osób
    #i nadaje każdemuy numerek

    result_series = df_copy.groupby('Group')[target_col].mean() #srednia dla każdej grupy z tych czterech, zwraca kolumnę
    result_df = result_series.to_frame() #zwraca dataframe
    result_df.rename(columns={target_col: f"Średnia {target_col}"}, inplace=True) #zmiana nazw

    return result_df

def corr_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:

    #jeśli argument cols nie został podany, liczmymy dla wszystkich kolumn liczbowych
    if cols is None:
        cols = df.columns.tolist()

    # sprawdzenie poprawności kolumn
    for col in cols:
        if col not in df.columns:
            raise InvalidDataError(f"Brak kolumny '{col}' w DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidDataError(f"Kolumna '{col}' nie zawiera wartości liczbowych.")

    # obliczenie macierzy korelacji
    corr = df[cols].corr()

    return corr