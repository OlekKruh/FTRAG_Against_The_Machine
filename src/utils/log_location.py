import inspect
from src.utils.exits import error_exit


def log_location() -> str | None:
    frame = inspect.currentframe()
    try:
        if frame and frame.f_back:
            caller_frame = frame.f_back
            code = caller_frame.f_code
            func_name = code.co_name

            self_obj = caller_frame.f_locals.get("self", None)
            class_name = self_obj.__class__.__name__ if self_obj else "StaticOrGlobal"

            del caller_frame

            return f">>> Location: {class_name}.{func_name}"
        else:
            error_exit(">>> Critical Error -> log_location")
    finally:
        del frame
