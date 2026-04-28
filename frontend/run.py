from __future__ import annotations

import pathlib
import sys


def _ensure_project_root_on_path() -> None:
    current = pathlib.Path(__file__).resolve()
    project_root = current.parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _ensure_project_root_on_path()
    from microfluidic_control_system.frontend.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()

