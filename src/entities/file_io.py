import json
from pathlib import Path
from typing import Any, cast
from pydantic import BaseModel
from src.utils.exits import error_exit
from src.utils.log_location import log_location


class FileIO:
    @staticmethod
    def file_read(path: Path) -> str:
        """
        Reads and returns the entire text content of a file.

        Args:
            path (Path): The absolute or relative path to the file.

        Returns:
            str: The raw text content of the file.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw_text: str = file.read()
                return raw_text
        except FileNotFoundError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: The file '{path}' was not found.")
        except PermissionError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: No permission to read '{path}'.")
        except UnicodeDecodeError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Encoding issue. Use UTF-8 for '{path}'.")
        except Exception as e:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Unexpected error during parsing: {e}")

    @staticmethod
    def file_read_json[BaseModelT: BaseModel](path: Path, obj: type[BaseModelT]) -> BaseModelT:
        """
        Reads a JSON file and validates it against a specified Pydantic model.

        Args:
            path (Path): The path to the JSON file.
            obj (type[BaseModelT]): The Pydantic BaseModel class to validate against.

        Returns:
            BaseModelT: An instance of the provided Pydantic model populated with the JSON data.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return cast(BaseModelT, obj.model_validate(data))

        except FileNotFoundError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: The JSON file '{path}' was not found.")
        except PermissionError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: No permission to read '{path}'.")
        except UnicodeDecodeError:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Encoding issue. Use UTF-8 for '{path}'.")
        except json.JSONDecodeError as e:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Invalid JSON format in '{path}'. Error: {e}")
        except Exception as e:
            error_exit(f"{log_location()}\n"
                       f">>> Context: Error parsing or validating JSON for model {obj.__name__}: {e}")

    @staticmethod
    def file_write_json(dir_path: Path, file_name: str, data: Any) -> None:
        """
        Serializes data and writes it to a JSON file, creating directories if necessary.

        Automatically converts Pydantic BaseModels or lists of BaseModels into
        dictionaries before saving.

        Args:
            dir_path (Path): The directory path where the file will be saved.
            file_name (str): The name of the output JSON file.
            data (Any): The data to serialize (BaseModel, list, or primitive types).
        """
        if isinstance(data, BaseModel):
            ready_data = data.model_dump()
        elif isinstance(data, list):
            ready_data = [item.model_dump() if isinstance(item, BaseModel) else item for item in data]
        else:
            ready_data = data

        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        end_path = dir_path / file_name

        try:
            with open(end_path, "w", encoding="utf-8") as file:
                json.dump(ready_data, file, indent=4)
                print(f">>> File: '{file_name}' saved successfully\n"
                      f">>> Location: '{dir_path}'")
        except Exception as e:
            error_exit(f"{log_location()}\n"
                       f">>> {e}")
