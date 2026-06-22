from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_classical_manifest import main as run_classical_manifest_main

def run_mice() -> None:
    """Run mice."""
    forwarded_args = list(sys.argv[1:])
    if "--method_keys" not in forwarded_args:
        forwarded_args = ["--method_keys", "mice"] + forwarded_args

    sys.argv = [sys.argv[0]] + forwarded_args
    run_classical_manifest_main()


if __name__ == "__main__":
    run_mice()
