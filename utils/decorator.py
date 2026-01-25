import time
import functools


def measure_time(func): # Prints the execution time of the function

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-precision start time

        # Call the original function
        result = func(*args, **kwargs)

        end_time = time.perf_counter()  # End time
        run_time = end_time - start_time

        print(f"|--- Execution time of '{func.__name__}': {run_time:.4f} s")

        return result

    return wrapper