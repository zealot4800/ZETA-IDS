# ZETA-IDS / AEGIS

Evidence-guided intrusion detection with a tree-based data plane, local Ollama
reasoning, training-evidence corroboration and adaptive pattern memory.

## Setup

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-baselines.txt
```

Serve `zeta-ids:qwen3-8b` in Ollama. Configure dataset paths in
`config/datasets.yaml`, AEGIS in `config/controller.yaml`, and baselines in
`config/baselines.yaml`.

## Run

```bash
# AEGIS
scripts/run_pipeline.sh --method aegis --dataset all --all-flows

# RLFE-IDS
scripts/run_pipeline.sh --method rlfe_ids --dataset all --all-flows

# SCALA-NIDS
scripts/run_pipeline.sh --method scala_nids --dataset all --all-flows

# MA-IDS
scripts/run_pipeline.sh --method ma_ids --dataset all --all-flows

# SERA-IDS
scripts/run_pipeline.sh --method sera_ids --dataset all --all-flows
```

Datasets: `kdd`, `cicids2018`, `ton_iot`, `unsw_nb15`.
Baselines: `rlfe_ids`, `scala_nids`, `ma_ids`, `sera_ids`.
`--all-flows` processes the full replay; construction uses the configured limit.
Run each architecture's command separately, waiting for completion before
starting the next. Each command processes KDD, CICIDS2018, TON-IoT and
UNSW-NB15 sequentially and stops on the first failure. Replace `--dataset all`
with a dataset name to run just that dataset. Run these commands from the
`ZETA-IDS` directory, or use the absolute path to `scripts/run_pipeline.sh`.

Preview a selected architecture without inference:

```bash
scripts/run_pipeline.sh --method rlfe_ids --dataset all --dry-run
```

AEGIS results: `results/runs/`. Baseline results: `results/baselines/`.
Each run saves its configuration, decisions, metrics and completion status.
Export baseline summaries with:

```bash
.venv/bin/python scripts/summarize_baselines.py > /tmp/baseline-results.csv
```

`OLLAMA_BASE_URL` overrides the server; `OLLAMA_BASELINE_MODEL` overrides the
baseline model. Launchers accept `ZETA_NATIVE_THREADS` and `ZETA_CPU_AFFINITY`.

Baselines are paper adaptations, not exact author reproductions. SERA-IDS uses
training-derived confidence proxies. SCALA retains source-validation and
adaptation gates. Replay labels are used only after prediction/update for
metrics. Match training cohorts and budgets before comparing against AEGIS;
current experiments do not hold out attack families for zero-day evaluation.
