import time
import functools


def measure_time(func): #drukuje czas działania funkcji

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Precyzyjny czas rozpoczęcia

        # Wywołanie oryginalnej funkcji
        result = func(*args, **kwargs)

        end_time = time.perf_counter()  # Czas zakończenia
        run_time = end_time - start_time

        print(f"|--- Czas wykonania '{func.__name__}': {run_time:.4f} s")

        return result

    return wrapper