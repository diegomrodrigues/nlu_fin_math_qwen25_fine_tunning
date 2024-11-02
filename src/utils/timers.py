import time

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        print(f'{self.name} took {self.duration:.2f} seconds') 