"""Run ZETA-IDS or the paper method on shared leave-one-attack-out folds."""

import argparse
from pathlib import Path

from .loao import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/loao.yaml"))
    parser.add_argument("--dataset")
    parser.add_argument("--attack")
    parser.add_argument("--method", choices=("zeta", "paper", "both"),
                        help="Explicit comparison workflow; omitted keeps the original ZETA-IDS command")
    args = parser.parse_args()
    run(args.config, args.dataset, args.attack, args.method)


if __name__ == "__main__":
    main()
