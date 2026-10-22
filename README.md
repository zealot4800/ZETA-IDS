# ZETA-IDS and reference-paper leave-one-attack-out evaluation

The default ZETA-IDS workflow evaluates four datasets independently. For each configured attack family, it removes that family from model fitting and source validation, then evaluates the fixed held-out test flows. It reports one conditional policy (C), built from an attack-only tree, source-certified benign regions, a source-selected membership gate, and conditional benign subregions.

`BENIGN` is an accepted flow, `KNOWN_ATTACK` is an attack decision, and `UNRESOLVED` is an escalation candidate. Bypass is the fraction of held-out attack flows accepted as `BENIGN`; unresolved flows are reported separately.

## Run

```
Dataset availbility at: "https://zenodo.org/records/23553819"
```

Install the packages in `requirements.txt`, then run from the repository root:

```bash
python3 -m src
python3 -m src --dataset cicids2018 --attack 'Port Scanning'
```

The command prints one C result per held-out family to the terminal and writes no reports or artifacts. It prints bypass, attack decisions, unresolved counts, benign acceptance, and known-attack recall. For CICIDS2018, it also prints bypass counts for each original label within the held-out family, including distinct feature patterns. Settings and dataset paths are in `config/loao.yaml`. The `benign_acceptance.max_fraction` setting caps source-calibration benign acceptance at 5% by retaining only the flows closest to a certified benign-region center. This gate can only turn a `BENIGN` decision into `UNRESOLVED`; it does not change attack decisions or use held-out test labels. Test benign acceptance is reported separately because distribution shifts can make it differ from the source-calibration limit.

The implementation is split across `src/loao_data.py` (loading and partitions), `src/loao_model.py` (attack tree and benign-region fitting), `src/loao_policies.py` (membership and conditional policy selection), and `src/loao.py` (fold evaluation and terminal output). Dataset-specific splits and tunable experiment settings live in `config/loao.yaml`.

## CICIDS2018 input

The CICIDS2018 input is `data/CICIDS2018.pkl`. It has 250,000 rows, ten mapped classes, and 125,000 benign plus 125,000 attack flows. The other inputs are `data/UNSW_NB15.pkl`, `data/TON-IOT.pkl`, and `data/KDD.pkl`.

## KDD input

`data/KDD.pkl` contains 148,517 NSL-KDD records with 41 traffic features. Its 125,973 `train` rows have detailed attack names; its 22,544 `test` rows have only `Normal` or `Attack`. KDD leave-one-attack-out evaluation uses **only the detailed-label train rows**, with feature-group-disjoint train, validation, and held-out partitions. The 22,544 coarse-label rows remain in the original PKL and are explicitly excluded from this experiment. Results from these partitions are not results on the supplied KDD `test` split.

The 22 detailed train attack names are mapped deterministically to four families—DoS, Probe, R2L, and U2R—using the [original KDD attack taxonomy](https://kdd.ics.uci.edu/databases/kddcup99/training_attack_types). The exact mapping, selected split, and row-count checks are in `config/loao.yaml`. All 41 traffic features are retained; `difficulty`, labels, and `split` are excluded from model inputs. Run `python3 -m src --dataset kdd` to evaluate its four families. U2R has only 52 selected rows, including 12 held-out test rows, so its fold has limited support.

CICIDS2018 is a class-aware research sample, so its class frequencies do not represent deployment prevalence. Repeated flows are correlated; row-level confidence bounds are nominal and should not be read as guarantees for independent captures. The CICIDS2018 PKL contains flow summaries but no endpoint identifiers or timestamps for sequence-level scan analysis.


## Independent unresolved-flow controller

The optional controller follows this routing:

```text
First layer: BENIGN / KNOWN_ATTACK -> existing decision
First layer: UNRESOLVED -> Jev binary detection
    BENIGN -> controller benign
    UNCERTAIN / ERROR -> unresolved
    ATTACK -> OpenRouter investigation -> family / UNKNOWN_ATTACK / INSUFFICIENT_EVIDENCE
```

`src/controller.py` owns routing, request budgets, calibration and the CLI;
`src/jev.py` uses the official TypeSafe SDK; `src/agent.py` implements the bounded
tool loop; `src/evidence.py` builds source references and read-only evidence.
All controller prompts, provider settings, limits, prior attack descriptions and
output settings are in `config/controller.yaml`. First-layer settings stay in
`config/loao.yaml`.

The controller reuses `fit_fold` to prepare the selected existing LOAO model once,
then freezes that model for the entire controller run. It never changes its tree,
rules, regions or thresholds. The existing LOAO CLI and evaluation metrics remain
available. No detector models or extra datasets are saved.

Install the dependencies into the project environment:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

The Jev integration follows the [official JevDetector reference](https://github.com/jev-sec/jev-ids/blob/main/jev_ids/detectors/jev.py)
and asks only its binary `noul` question. The default pins Jev 1.14.0; model changes
require fresh calibration. Stage 2 defaults to Claude Opus 4.6 through the
[OpenRouter tool-calling API](https://openrouter.ai/docs/guides/features/tool-calling).
Changing an OpenRouter model requires only configuration; select a compatible
`response_format` (`json_schema`, `json_object`, or `none`). Local schema and
observation-value validation always apply.

Preview requests and validate a real dataset schema without credentials, model
fitting, API calls, or output files:

```bash
.venv/bin/python -m src.controller --dataset kdd --attack Probe --max-flows 2 --dry-run
.venv/bin/python -m src.controller --config config/controller.yaml --dataset cicids2018 --attack 'Port Scanning' --dry-run
```

Dry runs use a clearly marked schema-only query with null feature values and
actual source-training examples. They do not pretend to have classified an
unresolved flow. Set `general.dry_run_show_requests` to inspect those requests.

For real API use, export `TYPESAFE_API_KEY` and `OPENROUTER_API_KEY` in the process
environment and set `general.enabled: true`. Their environment-variable names are
configurable. The program does not read `.env` automatically or log credentials.
Use [`.env_example`](.env_example) as the template for your local `.env`; it
includes instructions for loading the variables into your shell. Commit only
the template, keeping real credentials in the ignored `.env` file.

### Source calibration

The default `jev.calibration.mode: required` uses a matching source-calibration
artifact. Without one, every selected unresolved flow stays `UNCERTAIN` with
`controller_status: UNCALIBRATED`, and no model requests are sent. Only explicit
`mode: experimental` enables the configured unvalidated probability thresholds;
records then carry `calibration_status: experimental`.

Calibration samples only source-validation flows that the frozen first layer
left unresolved. It removes duplicate feature patterns and ambiguous patterns,
selects lower and upper thresholds on source tuning rows, then checks the frozen
thresholds once on a disjoint source certification partition. The configured
budgets bound attack-to-benign and benign-to-attack errors with separate one-sided
exact binomial bounds. They describe source certification; they do not guarantee
performance on a previously unseen attack or shifted target population.

Calibration is an explicit paid operation. The default samples require up to
1,400 Jev requests per dataset and held-out fold: 200 benign plus 200 attack flows
for threshold selection, then 500 of each for certification. Both calibration and
ordinary controller commands have a **$10 budget per CLI invocation**, configured
by `budgets.max_cost_usd`. There is no request-count cap. Each request is attempted
once. Calibration that cannot finish within its dollar budget stays uncertified;
small source classes or invalid predictions can also prevent certification.

```bash
.venv/bin/python -m src.controller --dataset kdd --attack Probe --calibrate
```

The artifact is written below the configured results directory, for example
`results/controller/kdd/Probe/jev_calibration.json`. It is bound to the dataset,
fold, code, first-layer configuration, reference data, Jev model and prompts.
Changes invalidate it. Existing artifacts are not overwritten; use another
`--output-dir` for a fresh calibration and the subsequent run.

After successful calibration, process a small number of actual unresolved flows:

```bash
.venv/bin/python -m src.controller --dataset kdd --attack Probe --max-flows 5
.venv/bin/python -m src.controller --config config/controller.yaml --dataset kdd --attack Probe --max-flows 5 --output-dir results/controller
```

### Evidence, output and limits

Only original permitted feature columns enter inference. Query labels, family
IDs, held-out family names, dataset IDs, row indices, and target statistics stay
out of model requests and tools. Binary labels appear only on authorized
source-training few-shot examples. Source baselines use benign training rows.
Provider adapters and investigation tools never receive the complete labeled test frame.

The tools expose flow observations, source-training comparisons and configured
prior knowledge. Reported observations must match supplied feature values.
Specific families require supporting observations; missing packet or temporal
information remains unavailable. Evidence strength is qualitative, not a
calibrated confidence probability. Structured validation cannot prove the truth
of an LLM's interpretation, so family reports still require appropriate review.

Real runs save validated `flows.jsonl` and `summary.json` under a unique run
directory inside `results/controller/`, which is ignored by Git. Results contain
one record per selected unresolved flow. Jev benign/uncertain/error records have
null investigation fields. Unknown or insufficient family evidence and advanced
API errors preserve Jev's binary ATTACK decision. Summary counts describe
controller routing, not ground-truth accuracy; target labels are not used here.

Jev and OpenRouter share the same $10 allowance within a controller run. A new CLI
invocation starts a fresh allowance; calibration and a later evaluation therefore
have separate $10 limits. The flow limit and investigation step/tool limits still
bound the work, while request counts are recorded only for reporting. Identical
requests reuse in-memory results without another charge.

Before each request, the controller checks a conservative input-size estimate plus
the maximum output allowance against the remaining budget. It stops before a
request that cannot fit, or if cost accounting becomes unavailable. OpenRouter's
reported `usage.cost` is used when present; otherwise costs are calculated from
reported tokens and configured rates. Model responses are still validated, and
failed requests are recorded without retrying.

The configured rates are $0.042 input / $0 output per million tokens for
[Jev](https://typesafe.ai/) and $5 input / $25 output for
[Claude Opus 4.6](https://openrouter.ai/anthropic/claude-opus-4.6), checked on
2026-10-21. OpenRouter requests also restrict provider token prices to these
configured rates. Update the pricing entries when changing models.

The dollar limit is an application-side estimate, not a provider-enforced billing
cap. Taxes, account fees and usage from other programs are outside this ledger.
No full experiment or live model call is performed by installation or dry run.

## Two-method research comparison

The research CLI selects exactly two approaches: **ZETA-IDS** (`zeta`) and the
**feature-influence reconstruction + model uncertainty method** (`paper`) from
[Fawaz et al., IEEE TNSM 2026](https://doi.org/10.1109/TNSM.2026.3731401).
`both` runs those two methods; it is not another detector. The Jev/LLM controller
remains independent and is never invoked by a comparison.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src --method zeta --dataset cicids2018 --attack 'Port Scanning'
.venv/bin/python -m src --method paper --dataset cicids2018 --attack 'Port Scanning'
.venv/bin/python -m src --method both --dataset cicids2018 --attack 'Port Scanning'
```

Omit `--attack` to evaluate all configured families. Omit `--dataset` to use
`comparison.datasets`. Commands **without `--method` retain the original ZETA-IDS
terminal-only workflow**, including their existing settings and decisions.
Explicit `--method` commands use `comparison` settings and write comparison
reports when `comparison.save_results` is true. Full experiments run only when
explicitly requested through these commands.

`src/paper.py` contains the entire reference method. `src/loao.py` loads each
selected dataset once and runs both methods on identical existing test flow
identities. All experiment settings, including the paper backend, are in
`config/loao.yaml`; controller files and settings are unchanged. The RF and
XGBoost choices are classifier backends of the same paper method, not extra
research approaches. The default backend is RF. TensorFlow and XGBoost use CPU
packages, and the reference dependencies are loaded only when needed.

### Verified method and reproduction boundaries

The implementation was checked against the supplied complete paper and the
[author implementation](https://github.com/hussein-fawaz/ZDA-SHAP-UQ), commit
`88fa3c196a62f4621cb2f3081fa4fc4dc9979631`:

1. Fit a multiclass classifier on known **attack** source-training rows only.
   Exclude the withheld family from classifier fitting, SHAP selection,
   autoencoder fitting, normalization, GMM selection and threshold calibration.
   RF uses 400 trees; XGBoost uses `multi:softprob`. XGBoost's numeric settings
   expose its author-used library defaults rather than claiming a published
   hyperparameter search.
2. Use TreeSHAP for the **predicted class**, giving one signed attribution per
   classifier input feature. Select the smallest ranked set covering 90% of mean
   absolute training SHAP influence. The author code uses approximate forest
   SHAP and exact XGBoost SHAP; these defaults are retained. Standardize selected
   SHAP columns using source training only.
3. Reconstruct these vectors with the published symmetric dense autoencoder:
   encoder `256→128→64→32→16`, mirrored decoder, batch normalization, ReLU,
   10% hidden dropout and tanh output. Optimize MSE with Adam, batches of 512,
   at most 10 epochs and early stopping (patience 3 from the author code).
4. Split trees (RF) or boosting rounds (XGBoost) into 20 disjoint subensembles.
   Compute epistemic uncertainty as `H(mean probabilities) − mean(H(probabilities))`
   with natural logarithms, following equations (2)–(4). RF averages tree
   probabilities as in the author code; XGBoost applies softmax to each group's
   margins. This does not substitute entropy of a single prediction for epistemic
   uncertainty.
5. Form `[SHAP reconstruction MSE, epistemic uncertainty]`. As in the author
   implementation, standardize these two coordinates on known source-validation
   rows, fit GMM candidates with 1–6 components and full/diagonal/tied/spherical
   covariance, and select the lowest validation BIC. Score with negative log
   density and flag scores **strictly above the 95th validation percentile**.
   Numerically invalid or nonconverged GMM candidates cannot be selected.

The reproduction status is **`adapted_reproduction`**, not an exact replication
of the paper's published numbers. Differences are saved with every experiment:

- This repository's CICIDS2018, TON-IOT, UNSW-NB15 and KDD PKLs and existing
  feature-group-disjoint partitions replace the paper's CICIoT2023, NF–TON–IoT
  and CIC–DDoS2019 data and row-based splits. TON-IOT is not claimed to be the
  same release as NF–TON–IoT. Each invocation uses one configured seed, rather
  than the paper's five-run average.
- Original records and duplicates stay in their existing partitions. Missing
  numeric values use the existing training-median imputer, and categorical
  values use training-fitted one-hot encoding. The author code instead drops
  missing rows and duplicates. Variance cutoff 0.025, absolute skew cutoff 50,
  5% tail winsorization and correlation cutoff 0.93 come from its code. Every
  fitted preprocessing statistic here uses known source-training rows only;
  the author code filters features before splitting. Input standardization
  follows the paper text, although its classifier code does not apply it.
- The paper's 16-unit bottleneck and dropout at every hidden layer take priority
  over the author code's 8-unit bottleneck and selective dropout. SHAP values
  use the author's StandardScaler and are **not clipped to tanh's output range**.
  Adam and batch-normalization defaults are exposed explicitly in configuration.
- Autoencoder early stopping uses a deterministic group-disjoint 80/20 split
  within source training. The author's code reuses the outer validation set;
  here that set is reserved for the GMM and novelty threshold. As in that code,
  GMM fitting, BIC selection and the percentile use the same source-validation
  population; this is not an independent false-positive certification.
- This comparison reports positive-class zero-day F1, whereas the paper also
  reports macro F1. Explicit row limits or a SHAP selection-row limit produce
  a reduced-data experiment and are recorded in metadata. Default limits are
  null, so no rows are sampled out of the existing partitions.

`paper.resources` controls CPU threads, batch size, dtype and the maximum dense
array size. The memory guard applies to individual arrays, not the entire Python
process. Inference is batched and has no fitting or calibration steps. ZETA-IDS's
existing algorithms and parameters are preserved.

### Metrics and files

The **primary population is attack-only**, including known attacks and the
withheld family. The withheld family is the positive class. ZETA-IDS `UNRESOLVED`
is treated as an investigation candidate solely for this comparison; unresolved
traffic does **not** establish maliciousness or a genuine zero-day. Paper
`ZERO_DAY_CANDIDATE` is its native novelty decision. ZETA-IDS `BENIGN` and
`KNOWN_ATTACK` remain their native decisions, and neither counts as a positive
novelty prediction.

Precision divides true withheld candidates by all candidates; recall divides
them by all withheld attacks. Known false-novelty rate divides known candidates
by all known attacks. Known classification accuracy scores the underlying family
prediction on **all known attacks**, regardless of routing; correctly routed
accuracy additionally requires `KNOWN_ATTACK`. All denominators and TP/FP/FN/TN
counts are included. AUROC and noninterpolated average precision (AUPRC) use the
paper's continuous negative log density. These metrics are unavailable for ZETA
because it supplies no continuous novelty score.

ZETA-IDS benign acceptance, benign escalation, known-attack recall, and attack-to-
benign bypass are reported separately using the original full-traffic metric
implementation. Latency and throughput in the comparison measure the same
attack-only batch population, including each method's preprocessing. Supplemental
benign inference and reporting are excluded. Memory is the **cumulative process
peak RSS**, not a separately measured model footprint.

Each explicit comparison creates a fresh timestamped directory under
`results/comparison/` containing:

- `metrics.csv`: detailed metrics per dataset, withheld family and method.
- `summary.json`: the side-by-side comparison table and detailed metrics.
- `predictions.csv`: original positional row IDs, test split IDs, truth labels
  for evaluation, native decisions, predicted known families and raw paper
  scores/MSE/uncertainty. ZETA benign rows are marked as supplemental.
- `experiment_metadata.json`: full experiment configuration, package versions,
  dataset checksums, source-selection rules, feature schemas, class mappings,
  split row-counts/hashes, selected SHAP features, GMM settings and reproduction
  deviations. Failures are marked explicitly; completed earlier folds remain.

Existing experiment directories are never overwritten. No trained models,
additional datasets, baselines, ablation variants or visualizations are saved.
