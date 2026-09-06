import sys
from typing import NoReturn


def error_exit(message: str) -> NoReturn:
    """
    Terminates the program execution with an error code (1).

    Prints a custom error message to the console before exiting.
    Used to handle critical failures, such as invalid command-line
    arguments or file reading errors.

    Args:
        message (str): The text describing the cause of the error.
    """
    print(f">>> Error Exit\n"
          f"{message}")
    sys.exit(1)


def warning(message: str) -> None:
    """
    Logs a non-critical warning message to the console.

    Used to inform the user about non-fatal issues, such as missing
    or invalid configuration keys that are being replaced by safe default values.

    Args:
        message (str): The text describing the warning.
    """
    print(f">>> Warning\n"
          f"{message}")


def norm_exit() -> NoReturn:
    """
    Terminates the program execution gracefully with a success code (0).

    Prints a farewell message to the console before exiting. Used when the
    application is closed intentionally without any errors.
    """
    print("Program closed. Have a nice day.")
    sys.exit(0)
