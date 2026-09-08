import inspect
from src.utils.exits import error_exit


def log_location() -> str | None:
    """
    Determines the class and function name of the caller.

    Uses the inspect module to step back one frame in the call stack
    and extract the execution context. If called inside a class method,
    it identifies the class name via the 'self' reference.

    Returns:
        str: A formatted string indicating the caller's location
             (e.g., '>>> Location: MyClass.my_method').
    """
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
