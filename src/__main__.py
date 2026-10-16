"""Run the explicit ZETA-IDS simulation command."""

import argparse
from pathlib import Path

from . import agent
from .simulation import limits, load_configuration, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-guided progressive traffic replay")
    parser.add_argument("command", choices=("simulate",))
    parser.add_argument("--dataset", choices=("cic2017", "cic2018", "both"))
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--backend", choices=("mock", "openrouter"))
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    system, investigator = load_configuration()
    if args.dataset is not None and args.dataset != "both":
        system["dataset"] = args.dataset
    if args.sample_ratio is not None:
        if not 0 < args.sample_ratio <= 1:
            parser.error("--sample-ratio must be in (0, 1]")
        system["sampling"]["ratio"] = args.sample_ratio
    if args.backend is not None:
        investigator["provider"] = args.backend
    datasets = ("cic2017", "cic2018") if args.dataset == "both" else (system["dataset"],)
    if args.dataset == "both" and any(Path(system["datasets"][name]["memory"]).exists() for name in datasets):
        parser.error("Both memory output paths must be unused")
    for name in datasets:
        system["dataset"] = name
        ledger = agent.Ledger(agent.Limits(**limits(investigator)))
        output = simulate(system, investigator, allow_paid=args.allow_paid, ledger=ledger)
        print(f"Pattern memory saved to {output}")


if __name__ == "__main__":
    main()
