"""Run the conditional-policy leave-one-attack-out experiment."""

import argparse
from pathlib import Path

from .loao import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/loao.yaml"))
    parser.add_argument("--dataset")
    parser.add_argument("--attack")
    args = parser.parse_args()
    run(args.config, args.dataset, args.attack)


if __name__ == "__main__":
    main()
