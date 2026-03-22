from __future__ import annotations

import sys
from pathlib import Path


def main():
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from train import main as run_pipeline

    run_pipeline()


if __name__ == "__main__":
    main()
