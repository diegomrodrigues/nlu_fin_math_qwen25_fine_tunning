import sys
from io import StringIO
from contextlib import contextmanager
from .timers import Timer

class OutputCapture:
    """Manages the capture of stdout during code execution."""
    
    @staticmethod
    @contextmanager
    def capture_stdout():
        stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout
        try:
            yield stdout
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def execute_code(code_block: str) -> str:
        """Executes a block of code and returns its output."""
        with Timer("Code execution"):
            with OutputCapture.capture_stdout() as output:
                try:
                    local_vars = {}
                    exec(code_block.strip(), {}, local_vars)
                    return output.getvalue().strip()
                except Exception as e:
                    print(f"Code execution error: {str(e)}")
                    return f"Error: {str(e)}" 