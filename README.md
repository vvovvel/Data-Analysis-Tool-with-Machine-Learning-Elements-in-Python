# Narzędzie do analizy danych z elementami uczenia maszynowego w Pythonie

Projekt implementuje modularne i elastyczne narzędzie do analizy danych (Data Analysis Tool) wykorzystujące zaawansowane techniki programowania obiektowego (OOP) w Pythonie, ze szczególnym uwzględnieniem modelowania predykcyjnego z użyciem biblioteki Scikit-learn.

### Cel Projektu

Głównym celem jest zademonstrowanie umiejętności w zakresie:
* **Modularnej Dekompozycji** kodu na logiczne klasy i pakiety.
* **Programowania Obiektowego** (Dziedziczenie, Abstrakcja) oraz **Wzorców Projektowych**.
* **Efektywnego Użycia** kluczowych bibliotek analitycznych (Pandas, Scikit-learn, Matplotlib).
* **Obsługi Błędów** (własne wyjątki i walidacja danych).

***

## ⚙️ Wymagania i Instalacja

Projekt wymaga środowiska **Python 3.9+** oraz następujących bibliotek analitycznych i pomocniczych:

* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `numpy`

📦 Instalacja
1. Klonowanie repozytorium
git clone https://github.com/twoje_repo/projekt.git
cd projekt

2. Instalacja zależności
pip install -r requirements.txt

3. Uruchomienie
python main.py


## 📦 **Struktura Projektu i Opis Modułów**

Projekt opiera się na **separacji obowiązków**, gdzie każda domena ma swój własny pakiet.

| Pakiet / Moduł | Główna Odpowiedzialność |
| :--- | :--- |
| **`main.py`** | **Sterowanie pipeline'em**, **Wybór danych do analizy**, Obsługa błędów. |
| **`data/`** | **Ładowanie** i **Walidacja** danych, **Uzupełnianie wartości NA**. |
| **`analysis/`** | **Obliczenia statystyczne** oraz **Wizualizacja** przy pomocy wykresów **scatter**, **histogram**, **boxplot**. |
| **`ml/`** | Klasy reprezentujące **modele ML** (klasa bazowa i dziedziczące) oraz **przygotowywanie danych** do ich implementacji. |
| **`outputs/`** | **Folder do zapisu** wszystkich wygenerowanych **wykresów** i logów czasowych. |
| **`tests/`** | **Testy** jednostkowe statystyk i walidatora. |
| **`utils/`** | **Context Manager** logujący czas i **Decorator** wypisujący czas działania funkcji. |


## 📁 Szczegółowa struktura katalogów 
```
projekt/
├── analysis/
│ ├── plots.py          # Funkcje wizualizacji (scatter, boxplot, histogram)
│ └── statistics.py     # Funkcje obliczające statystyki
├── data/
│ ├── exceptions.py     # Klasa InvalidDataError
│ ├── loading_and_prep.py # Praca z danymi 
│ ├── Sleep_health_and_lifestyle_dataset.csv    #przykładowy dataset nr 1
│ └── student_exam_scores.csv       #przykładowy dataset nr 2
├── ml/
│ ├── BaseModel.py  # Klasa abstarakcyjna służąca do minimalizacji powtarzajacego się kodu
│ ├── ClassifierModel.py    # Klasyfikacja KNN
│ ├── ClusteringModel.py    # Klasteryzacja KMeans
│ ├── DataPreparer.py       # Preprocessing dla ML (skalowanie, One-Hot Encoding)
│ ├── ModelRunner.py        # Factory/Strategy, użycie @measure_time
│ └── RegressionModel.py    # Regresja Liniowa
├── outputs/                
├── tests/
│ ├── test_statistics.py    # Testy jednostkowe funkcji statystycznych
│ └── test_validator.py     # Testy jednostkowe walidacji danych 
├── utils/
│ ├── context_manager.py    # Klasa TimeLoggerContext
│ └── decorator.py          # Funkcja @measure_time
├── main.py
└── README.md 
```
## 🔧 Jak użyć własnego zbioru danych?

W przypadku tego projektu, aby uruchomić pipeline na własnym pliku CSV, należy skopiować jedną z gotowych funkcji pipeline (np. `test_pipeline_exams()` lub `test_pipeline_sleep()`) i podmienić w niej:

- ścieżkę do danych (`DATA_PATH`)  
- listy kolumn (`REQUIRED_COLUMNS`, `POSITIVE_COLUMNS`, `FILL_NA_COLS`)  
- kolumnę identyfikatora (`ID_COL`)  
- kolumny używane w statystykach, wykresach i modelach ML (`STATS_COLUMNS`, `SCATTER_X/Y`, `HISTOGRAM_X`, `BOXPLOT_X/Y`, `REGRESSION_FEATURES/TARGET`, `CLASSIFICATION_FEATURES/TARGET`, `CLUSTERING_FEATURES`)

Przykładowa zmiana:

```Plaintext
    def test_pipeline_exams():
    
        DATA_PATH = os.path.join('data', 'student_exam_scores.csv')
    
        REQUIRED_COLUMNS = ['student_id', 'hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score']
    
        POSITIVE_COLUMNS = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score']
    
        ID_COL = 'student_id'
    
        FILL_NA_COLS = []
    
        FILL_NA_VALUE = 0
    
        STATS_COLUMNS = ['hours_studied', 'sleep_hours', 'previous_scores', 'exam_score']
    
        GROUPED_MEAN_COL = 'previous_scores'
        GROUPED_MEAN_TARGET = 'exam_score'
    
        SCATTER_X = 'hours_studied'
        SCATTER_Y = 'exam_score'
    
        HISTOGRAM_X = 'exam_score'
    
        BOXPLOT_X = 'previous_scores'
        BOXPLOT_Y = 'exam_score'
    
        REGRESSION_FEATURES = ['hours_studied']
        REGRESSION_TARGET = 'exam_score'
    
        CLASSIFICATION_FEATURES = ['hours_studied']
        CLASSIFICATION_TARGET = 'exam_score'
    
        CLUSTERING_FEATURES = ['hours_studied', 'attendance_percent']
    
        try:
    
            df = perform_loading_and_prep(DATA_PATH, REQUIRED_COLUMNS, FILL_NA_COLS, FILL_NA_VALUE, POSITIVE_COLUMNS)
    
            summary_stats = run_summary_stats(df, STATS_COLUMNS)
            grouped_mean = run_grouped_mean(df, GROUPED_MEAN_COL, GROUPED_MEAN_TARGET)
            corr_matrix = run_correlation_matrix(df, STATS_COLUMNS)
    
            plot_corr_matrix(corr_matrix)
            plot_data(df, 'scatter', SCATTER_X, SCATTER_Y)
            plot_data(df, 'histogram', HISTOGRAM_X)
            plot_data(df, 'boxplot', BOXPLOT_X, BOXPLOT_Y)
    
            lin_model = run_regression(df, ID_COL, REGRESSION_TARGET, REGRESSION_FEATURES)
            lin_model.plot("linear_students")
    
            # knn_model = run_classification(df, ID_COL, CLASSIFICATION_TARGET, CLASSIFICATION_FEATURES)
            # knn_model.plot("knn_students")
    
            kmeans_model = run_clustering(df, ID_COL, CLUSTERING_FEATURES, 5)
            kmeans_model.plot("cluster_students")
    
    
        except InvalidDataError as e:
            print(f"Błąd w danych (InvalidDataError): {e}")
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd podczas działania programu: {e}")
```

### Co projekt zrobi automatycznie?

- ✔ **Walidacja danych** — sprawdzenie poprawności i kompletności  
- ✔ **Czyszczenie** — usuwanie/uzupełnianie wartości brakujących  
- ✔ **Skalowanie** — normalizacja cech numerycznych  
- ✔ **Kodowanie One-Hot** — przygotowanie zmiennych kategorycznych  
- ✔ **Analiza statystyczna** — opisowe metryki i wizualizacje  
- ✔ **Modele ML** — regresja, klasyfikacja i klasteryzacja  

Po zakończeniu działania pipeline'u wszystkie wyniki oraz wykresy zostaną zapisane w katalogu `outputs/`.


## ➡️ Dalszy Rozwój i Potencjalne Ulepszenia

Projekt został zaprojektowany modularnie, aby umożliwić jego rozbudowę w następujących obszarach:

* **Zwiększenie Odporności Pipeline'u (Soft Failures):** Modyfikacja runnerów ML (`ModelRunner.py`) w celu zapewnienia, że błąd podczas trenowania jednego modelu (np. klasyfikacji) nie przerywa całego procesu analizy, pozwalając na kontynuację kolejnych kroków (np. klasteryzacji).
* **Dodanie Interfejsu Użytkownika (UI):** Zaimplementowanie prostej aplikacji dla użytkownika końcowego (np. opartej na Streamlit lub Flask), która zastąpi obecne, statyczne wywołania funkcji w main.py bardziej dynamicznym wyborem analizy.
* **Rozszerzenie Zestawu Modeli ML:** Dodanie kolejnych algorytmów uczenia maszynowego.
* **Poprawa Wizualizacji (3D i Etykiety):**
    * Implementacja wykresów 3D dla analizy cech (np. trzech cech naraz).
    * Wizualizacja etykiet klastrów/klas na oryginalnych, **nieskalowanych** danych w celu lepszej interpretacji wyników.