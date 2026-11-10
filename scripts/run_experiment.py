"""Run one experiment from the terminal; every ExperimentConfig field is a --flag.

Example:
    python scripts/run_experiment.py --model hybrid --attn_pos after_stage1 --epochs 30
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

# Make `src` importable when the script is run as `python scripts/run_experiment.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig  # noqa: E402
from src.runner import run_experiment  # noqa: E402


def str_to_bool(value: str) -> bool:
    """Parse ``true``/``false`` style strings, so booleans can be set with ``--amp false``."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """One ``--<field>`` per ``ExperimentConfig`` field, with the field's type and default."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in fields(ExperimentConfig):
        arg_type = str_to_bool if f.type is bool else f.type
        parser.add_argument(f"--{f.name}", type=arg_type, default=f.default,
                            help=f"default: {f.default!r}")
    return parser


def main() -> None:
    cfg = ExperimentConfig(**vars(build_parser().parse_args()))
    result = run_experiment(cfg)
    print(f"{cfg.run_name}: best val acc {result.best_val_acc:.4f} | "
          f"test acc {result.test_acc:.4f} | params {result.n_params:,} | "
          f"train time {result.train_time_s:.0f}s")


if __name__ == "__main__":
    main()
