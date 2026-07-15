"""Filesystem I/O utilities."""

import shutil
from pathlib import Path


def remove_dir(path: str) -> None:
    """Recursively remove a directory and all its contents.

    Parameters
    ----------
    path : str
        Path to the directory to remove.

    Raises
    ------
    NotADirectoryError
        If ``path`` exists but is not a directory.
    """
    dir_path = Path(path)

    # Silently return if the directory does not exist.
    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return

    # Guard against accidentally deleting a file.
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    # Recursively delete the directory tree.
    shutil.rmtree(dir_path)
    print(f"Successfully removed directory: {dir_path}")
