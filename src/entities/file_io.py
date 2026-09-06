import json
from pathlib import Path
from typing import Any, cast
from pydantic import BaseModel
from src.utils.exits import error_exit
from src.utils.log_location import log_location


class FileIO:
    @staticmethod
    def file_read(path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw_text: str = file.read()
                return raw_text
        except FileNotFoundError:
            error_exit(f">>> {log_location()}\n"
                       f">>> Context: The file '{path}' was not found.")
        except PermissionError:
            error_exit(f">>> {log_location()}\n"
                       f">>> Context: No permission to read '{path}'.")
        except UnicodeDecodeError:
            error_exit(f">>> {log_location()}\n"
                       f">>> Context: Encoding issue. Use UTF-8 for '{path}'.")
        except Exception as e:
            error_exit(f">>> {log_location()}\n"
                       f">>> Context: Unexpected error during parsing: {e}")

    @staticmethod
    def file_read_json[BaseModelT: BaseModel](path: Path, obj: type[BaseModelT]) -> BaseModelT:
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
    def file_write(dir_path: Path, file_name: str, data: Any) -> None:
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
            error_exit(f">>> {log_location()}\n"
                       f">>> {e}")
