import time # getting current time
import os # managing file paths


class TimeLoggerContext:

    def __init__(self, task_name, filename="log.txt", output_dir="outputs"):
        self.task_name = task_name
        self.output_dir = output_dir

        # creating the path to the file
        self.filename = os.path.join(self.output_dir, filename)

        # creating the directory if it doesn't already exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.start_seconds = None
        self.end_seconds = None

    def __enter__(self):

        self.start_seconds = time.time() # measuring start time
        start_time_str = time.ctime(self.start_seconds) # converting time to string

        # Use the full path self.filename
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"[START] {self.task_name} - {start_time_str}\n") # writing start of the task to file

        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # potential error -> its type, value, and traceback (exact line of code, etc.)

        self.end_seconds = time.time() # measuring end time

        elapsed_time = self.end_seconds - self.start_seconds # task duration
        end_time_str = time.ctime(self.end_seconds) # converting time to string

        with open(self.filename, "a", encoding="utf-8") as f:
            # error handling
            if exc_type is not None:
                f.write(f"[ERROR] {self.task_name} - {exc_type.__name__}: {exc_val}\n")

            # writing end of the task to file
            f.write(f"[END] {self.task_name} - {end_time_str} | Duration: {elapsed_time:.4f}s\n")
            f.write("-" * 50 + "\n") # visual separator for the log entry

        return False