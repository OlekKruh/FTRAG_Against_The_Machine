from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class PathFinder:
    """
    Recursively scans the target directory for .py and .md files.

    Populates the path_map dictionary where keys are file extensions
    and values are lists of corresponding Path objects.
    """
    target_dir: Path = Path()
    path_map: Dict[str, list[Path]] = field(default_factory=dict)

    def pars_dir(self) -> None:
        for file_path in self.target_dir.rglob("*"):
            ext = file_path.suffix.lower()
            if file_path.is_file() and ext in (".py", ".md"):
                self.path_map.setdefault(ext, []).append(file_path)
