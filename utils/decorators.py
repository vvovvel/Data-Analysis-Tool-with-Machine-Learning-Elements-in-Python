from functools import wraps

def print_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"\nWynik funkcji '{func.__name__}':")
        print(result)
        return result
    return wrapper
