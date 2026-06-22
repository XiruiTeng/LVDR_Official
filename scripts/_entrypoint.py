from __future__ import annotations

import sys
from pathlib import Path


def run_cli(command: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from lvdr_official.cli import main

    sys.argv = [sys.argv[0], command, *sys.argv[1:]]
    main()
