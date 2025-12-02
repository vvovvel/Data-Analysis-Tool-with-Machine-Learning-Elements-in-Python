import pandas as pd
import os

# --- PARAMETRY ---
# Proszę zmienić ścieżkę, jeśli plik student_exam_scores.csv nie jest obok tego skryptu
INPUT_FILE = 'student_exam_scores.csv'
OUTPUT_FILE = 'student_exam_scores_graded.csv'
MAX_SCORE = 52  # Ustalony maksymalny wynik (52 punkty)
SCORE_COLUMN = 'exam_score'

# Definicja progów ocen (progi dla pd.cut muszą obejmować cały zakres)
# Używamy procentów, a następnie przeliczamy na punkty.
# [0%, 40%, 60%, 75%, 90%, 100%+]
# Progi: 0.0, 20.8, 31.2, 39.0, 46.8, 52.0 (plus mały margines na max)

# Progi w punktach (używamy float, aby być precyzyjnym):
BINS = [
    0.0,  # Poniżej 40% (0)
    MAX_SCORE * 0.40,  # 40% (20.8) - Granica D
    MAX_SCORE * 0.60,  # 60% (31.2) - Granica C
    MAX_SCORE * 0.75,  # 75% (39.0) - Granica B
    MAX_SCORE * 0.90,  # 90% (46.8) - Granica A
    MAX_SCORE + 1  # Powyżej max, żeby objąć wszystkie wyniki (53.0)
]

# Etykiety dla każdej z utworzonych grup (od najniższej do najwyższej)
LABELS = ['E', 'D', 'C', 'B', 'A']


def add_grades_column():
    print(f"Ładowanie danych z: {INPUT_FILE}")

    try:
        # Wczytanie danych
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(
            f"BŁĄD: Plik '{INPUT_FILE}' nie został znaleziony. Upewnij się, że jest w tej samej lokalizacji co skrypt.")
        return

    if SCORE_COLUMN not in df.columns:
        print(f"BŁĄD: W pliku CSV brakuje kolumny '{SCORE_COLUMN}'.")
        return

    # Użycie pd.cut do dyskretyzacji kolumny exam_score
    df['exam_grade'] = pd.cut(
        df[SCORE_COLUMN],
        bins=BINS,
        labels=LABELS,
        right=False,  # Przedziały są otwarte z lewej, zamknięte z prawej: (a, b]
        include_lowest=True  # Uwzględnia najniższą wartość
    )

    # Przekształcenie na typ 'object' (string), aby uniknąć problemów z kategoriami
    df['exam_grade'] = df['exam_grade'].astype(str)

    # Zapis nowego pliku
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n-------------------------------------------")
    print(f"Sukces! Nowa kolumna 'Exam_Grade' została dodana.")
    print(f"Plik wynikowy zapisano jako: {OUTPUT_FILE}")
    print(f"Teraz przenieś ten plik do folderu 'data/' w Twoim projekcie.")


if __name__ == "__main__":
    add_grades_column()