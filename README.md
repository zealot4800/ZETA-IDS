# ZETA-IDS leave-one-attack-out evaluation

This code evaluates four datasets independently. For each configured attack family, it removes that family from model fitting and source validation, then evaluates the fixed held-out test flows. It reports one conditional policy (C), built from an attack-only tree, source-certified benign regions, a source-selected membership gate, and conditional benign subregions.

`BENIGN` is an accepted flow, `KNOWN_ATTACK` is an attack decision, and `UNRESOLVED` is an escalation candidate. Bypass is the fraction of held-out attack flows accepted as `BENIGN`; unresolved flows are reported separately.

## Run

Install the packages in `requirements.txt`, then run from the repository root:

```bash
python3 -m src
python3 -m src --dataset cicids2018 --attack 'Port Scanning'
```

The command prints one C result per held-out family to the terminal and writes no reports or artifacts. It prints bypass, attack decisions, unresolved counts, benign acceptance, and known-attack recall. For CICIDS2018, it also prints bypass counts for each original label within the held-out family, including distinct feature patterns. Settings and dataset paths are in `config/loao.yaml`.

The implementation is split across `src/loao_data.py` (loading and partitions), `src/loao_model.py` (attack tree and benign-region fitting), `src/loao_policies.py` (membership and conditional policy selection), and `src/loao.py` (fold evaluation and terminal output). Dataset-specific splits and tunable experiment settings live in `config/loao.yaml`.

## CICIDS2018 input

The CICIDS2018 input is `data/CICIDS2018.pkl`. It has 250,000 rows, ten mapped classes, and 125,000 benign plus 125,000 attack flows. The other inputs are `data/UNSW_NB15.pkl`, `data/TON-IOT.pkl`, and `data/KDD.pkl`.

## KDD input

`data/KDD.pkl` contains 148,517 NSL-KDD records with 41 traffic features. Its 125,973 `train` rows have detailed attack names; its 22,544 `test` rows have only `Normal` or `Attack`. KDD leave-one-attack-out evaluation uses **only the detailed-label train rows**, with feature-group-disjoint train, validation, and held-out partitions. The 22,544 coarse-label rows remain in the original PKL and are explicitly excluded from this experiment. Results from these partitions are not results on the supplied KDD `test` split.

The 22 detailed train attack names are mapped deterministically to four families—DoS, Probe, R2L, and U2R—using the [original KDD attack taxonomy](https://kdd.ics.uci.edu/databases/kddcup99/training_attack_types). The exact mapping, selected split, and row-count checks are in `config/loao.yaml`. All 41 traffic features are retained; `difficulty`, labels, and `split` are excluded from model inputs. Run `python3 -m src --dataset kdd` to evaluate its four families. U2R has only 52 selected rows, including 12 held-out test rows, so its fold has limited support. If the original text inputs are available, `python3 scripts/convert_kdd.py` can rebuild the PKL.

CICIDS2018 is a class-aware research sample, so its class frequencies do not represent deployment prevalence. Repeated flows are correlated; row-level confidence bounds are nominal and should not be read as guarantees for independent captures. The CICIDS2018 PKL contains flow summaries but no endpoint identifiers or timestamps for sequence-level scan analysis.
