import time #pobieranie aktualnego czasu
import os #zarządzanie ścieżkami plików


class TimeLoggerContext:

    def __init__(self, task_name, filename="log.txt", output_dir="outputs"):
        self.task_name = task_name
        self.output_dir = output_dir


        self.filename = os.path.join(self.output_dir, filename) #tworzenie ścieżki do pliku

        os.makedirs(self.output_dir, exist_ok=True) #tworzenie pliku, chyba że on już istnieje to wtedy nie tworzymy go znowu

        self.start_seconds = None
        self.end_seconds = None

    def __enter__(self):

        self.start_seconds = time.time() #mierzymy czas startu
        start_time_str = time.ctime(self.start_seconds) #zamienianie czasu na string

        # Używamy pełnej ścieżki self.filename
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"[START] {self.task_name} - {start_time_str}\n") #zapis początku działania do pliku

        return self

    def __exit__(self, exc_type, exc_val, exc_tb): #ewentualny błąd -> jego typ, wartość i traceback -> gdzie dokłądnie wystąpił wyjątek, w której linii kodu itd

        self.end_seconds = time.time() #mierzymy czas zakończenia

        elapsed_time = self.end_seconds - self.start_seconds #czas trwania całego zadania
        end_time_str = time.ctime(self.end_seconds) #zamienianie czasu na string

        with open(self.filename, "a", encoding="utf-8") as f:
            #obsługa błędu
            if exc_type is not None:
                f.write(f"[ERROR] {self.task_name} - {exc_type.__name__}: {exc_val}\n")

            #zapis końca działania do pliku
            f.write(f"[END] {self.task_name} - {end_time_str} | Trwanie: {elapsed_time:.4f}s\n")
            f.write("-" * 50 + "\n") #ładny koniec jednej cześci zapisu

        return False