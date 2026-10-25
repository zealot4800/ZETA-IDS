"""Run ZETA-IDS or the paper method on shared leave-one-attack-out folds."""

import argparse
from pathlib import Path

from .loao import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/loao.yaml"))
    parser.add_argument("--dataset")
    parser.add_argument("--attack")
    parser.add_argument("--acceptance-sensitivity", action="store_true",
                        help="Offline source-certified benign-cap exploration; no controller/API calls")
    parser.add_argument("--method", choices=("zeta", "paper", "both"),
                        help="Explicit comparison workflow; omitted keeps the original ZETA-IDS command")
    args = parser.parse_args()
    if args.acceptance_sensitivity and args.method is not None:
        parser.error("--acceptance-sensitivity uses the first-layer workflow; omit --method")
    run(args.config, args.dataset, args.attack, args.method, args.acceptance_sensitivity)


if __name__ == "__main__":
    main()
