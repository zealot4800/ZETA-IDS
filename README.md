# ZETA-IDS

ZETA-IDS evaluates network intrusion detection with **leave-one-attack-family-out** folds. For each fold, it excludes one attack family from model fitting and source validation, then tests on fixed held-out flows. The default command runs the ZETA-IDS conditional policy (C). Two optional workflows provide an unresolved-flow controller and a comparison with a published novelty-detection method.

## Setup and data

Download the [datasets from Zenodo](https://zenodo.org/records/23553819) and place these files in `data/`:

| Dataset | File |
| --- | --- |
| CICIDS2018 | `CICIDS2018.pkl` |
| UNSW-NB15 | `UNSW_NB15.pkl` |
| TON-IOT | `TON-IOT.pkl` |
| KDD | `KDD.pkl` |

Install dependencies and run commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dataset paths, folds, and experiment settings are in [`config/loao.yaml`](config/loao.yaml).

## Default ZETA-IDS evaluation

```bash
.venv/bin/python -m src
.venv/bin/python -m src --dataset cicids2018 --attack 'Port Scanning'
```

The default command evaluates all configured families in all four datasets. It prints results to the terminal and saves no reports. `BENIGN` accepts a flow, `KNOWN_ATTACK` flags an attack, and `UNRESOLVED` sends it for further review. **Bypass** is the share of held-out attack flows accepted as `BENIGN`; unresolved flows are counted separately. The output also includes benign acceptance and known-attack recall.

### Work 1: benign acceptance gate

The internal detector retains BENIGN, KNOWN_ATTACK, and UNRESOLVED for research. Operational routing accepts only BENIGN; both other decisions go to the controller as UNRESOLVED. An attack-rule match is preserved as evidence and never counts as verified maliciousness.

`benign_acceptance.max_fraction: 0.05` is the reference baseline. The cap can restrict benign filtering; increasing it cannot recover flows rejected by region, membership, conditional, or attack-conflict checks. Its cutoff is now fitted on source tuning rows, not certification rows. The original attack tree, region learning, membership and conditional mechanisms remain in use.

Run one baseline fold or an exploratory cap sensitivity experiment (terminal output only; no APIs):

```bash
.venv/bin/python -m src --dataset cicids2018 --attack DDoS
.venv/bin/python -m src --dataset cicids2018 --attack DDoS --acceptance-sensitivity
```

The sensitivity settings are in `benign_acceptance.sensitivity` in `config/loao.yaml`: caps `[0.05, 0.10, 0.20, 0.40, 0.60, 0.80]`. It reserves a feature-group-disjoint portion of source validation before fitting the detector. All six caps share that detector and are fitted on its source tuning partition. The reserved partition is never used for the tree, regions, membership, conditional selection, or cap fitting. The withheld attack is excluded from all source partitions.

Candidates are checked against the unchanged source attack-leakage budget with simultaneous one-sided exact binomial bounds (Bonferroni adjustment over the six caps). Among certified, nondegenerate candidates, the recommendation maximizes tuning benign coverage, breaking ties toward the smaller cap. Test outcomes are evaluated only after that recommendation is frozen. Candidates with zero benign coverage are not successful optimizations. If none qualifies, no larger cap is approved and the reference configuration remains in place. A recommendation is exploratory: the command never changes the configured operating cap. The within-experiment 5% baseline shares the reserved split with the other candidates, so its numbers need not match the default baseline run.

Legacy three-way metrics remain internal-classifier diagnostics. New `gate_routing_json` metrics report accepted/escalated counts and percentages separately for all traffic, benign, known attacks, and withheld attacks. Overall escalation is one minus the fraction of **all flows** accepted as benign; benign acceptance and attack bypass use their own class denominators. Existing `controller_escalation_workload`, `benign_escalated`, and `overall_unresolved_*` fields retain their legacy internal-UNRESOLVED meaning; use `gate_*` fields for actual routing workload. Legacy detection-named metrics describe model decisions, not independent confirmation of maliciousness. Overlap and rejection diagnostics explain remaining workload, and inference latency is retained.

Work 1 changes neither the reference-paper method nor the LLM integrations. It makes no API calls during sensitivity evaluation. Provider migration and verifier/investigator behavior belong to Work 2 and Work 3.


**Dataset limits:** KDD folds use only the 125,973 detailed-label `train` records, split into group-disjoint training, validation, and held-out sets. The 22,544 coarse-label `test` records are excluded, so these results are not scores on the supplied KDD test split. KDD's 22 attack names map to DoS, Probe, R2L, and U2R; U2R has only 12 held-out test rows. CICIDS2018 is a class-aware research sample whose class frequencies do not reflect deployment prevalence. Repeated flows are correlated, and its flow summaries have no endpoint IDs or timestamps for sequence-level analysis.

## Controller architecture: audit and provisional design

The research hypothesis is that selective verification and investigation can resolve
traffic safely at manageable cost. No live improvement has been established.

### Audit and alternatives

The attack tree is trained on attacks only, so its family prediction cannot prove
that a flow is malicious. Benign regions use the same tree partitions; attack and
benign evidence are correlated. Rule matches, membership, conditional checks and
acceptance-cap checks therefore remain separate evidence fields.

The historical 5% benign acceptance cap deliberately limited useful filtering.
The controller now sets `routing.benign_acceptance_max_fraction: 1.0` to remove
that artificial cap; this does not guarantee acceptance or disable safety checks.
The standalone first-layer reference configuration retains its historical 5% cap. Overlapping
features, membership/conditional gates and the strict 0.1% leakage budget can reduce
acceptance further. The cap is already selected on tuning data, not its subsequent
certification rows. However, the former controller reused first-layer validation
rows to certify the verifier. Those observations were not independent of detector
fitting. The revised controller reserves separate source pools before fitting.
Existing local first-layer and paper algorithms remain unchanged.

| Candidate | Benefit | Required evidence / cost | Decision |
|---|---|---|---|
| A: benign-first | Avoids unvalidated direct attack alerts | May escalate many benign flows; needs independent benign-release checks | Provisional operational design |
| B: dual-evidence direct routing | Could avoid redundant verification of known attacks | Needs independent false-alert certification; tree predictions alone are insufficient | Evidence states implemented; direct alerts not authorized |
| C: risk-aware routing | Could skip uninformative calls and prioritize investigation | Useful additional evidence and source-supported alert policies are unmeasured | Uncertainty investigation is optional, with recommendations only |

A is currently the simplest defensible choice. B/C have no measured superiority or
certified direct-alert path. The code does not infer either from held-out test
performance. Required follow-up measurements include source benign retention,
attack leakage bounds, alert precision, unresolved workload, and paired LLM quality,
latency and cost. The existing offline `--acceptance-sensitivity` command explores
larger caps; it does not promote them into the controller or relax the leakage budget.

### Explicit state machine

`first_layer_evidence.evidence_state` distinguishes `ATTACK_SUPPORTED`,
`BENIGN_SUPPORTED`, `CONFLICTING`, and `INSUFFICIENT`. Benign support means region,
rule, membership and conditional support; operational acceptance additionally needs
the cap and independent gate authorization. Internal decisions remain available.

| Evidence / stage result | Security action | Investigation |
|---|---|---|
| Internal BENIGN, independently authorized gate | Accept; authorization recorded in fold gate audit | None |
| Internal BENIGN, uncertified gate | Escalate, never accept | Verify |
| Internal KNOWN_ATTACK or UNRESOLVED | Escalate; preserve evidence state | Verify |
| Certified verifier BENIGN | Accept under the categorical source policy | None |
| Certified verifier ATTACK | Alert | Attack-focused agent, or pending if disabled/unselected |
| Verifier UNCERTAIN | Retain unresolved | Optional uncertainty-focused agent |
| Missing/unsafe calibration, malformed output, request failure | Retain unresolved | No inferred security decision |
| Attack investigation identifies/abstains/fails | Preserve ATTACK/ALERT | Identified, unknown, insufficient, or error status |
| Uncertainty investigation recommends BENIGN or ATTACK | Retain unresolved | Recommendation only; no automatic release or alert |

The frozen upstream log separates `evidence_state`, `security_decision`,
`final_disposition`, `benign_release_policy`, `escalation_reason`, and verifier
recommendation. Investigation records join by `flow_id` and add their own status;
they never overwrite the upstream security decision. BENIGN outcomes do not reach
the investigator. Failed verification is not a successful malicious detection.

### Independent source assessment

Controller source validation is split by **identical model-input feature groups**:
50% for first-layer fitting/selection, 30% for an independent gate assessment,
5% for verifier tuning, and 15% for verifier certification. Original training/test
rows and the LOAO withheld-family exclusion remain unchanged. The first-layer
algorithm uses only the 50% source-validation pool; it never sees the other pools.

The frozen gate is assessed at the unchanged `known_attack_leakage_budget: 0.001`
with a one-sided exact binomial bound. Any accepted malicious member makes its
feature group a leakage error. An uncertified gate cannot accept benign traffic;
all its rows are escalated. The audit reports acceptance and errors separately, so
zero leakage from an all-abstaining detector is not claimed as useful resolution.

A CICIDS2018/DDoS diagnostic found that the former 40% fitting reserve left
only 2,475 attacks in the inner certification split. Even with zero errors its
Wilson upper bound was 0.1092%, above the unchanged 0.1% budget; the implementation
therefore disabled every benign region. Increasing fitting support to 50% provides
3,126 inner certification attacks and a zero-error upper bound of 0.08647%.
The independent outer gate then observed zero leaked attack groups out of 3,457,
with an exact upper bound of 0.08662%; it accepted 881 of 5,686 source benign rows.

After freezing this source-supported change, the DDoS fold's 37,498 test flows
included 18,750 benign, 13,343 known attacks and 5,405 held-out DDoS flows. The gate
accepted 2,870 benign flows (15.31%) and zero attacks, forwarding 34,628 flows.
This is one fold's measured result, not a maximum achievable coverage or a guarantee
for other folds/deployment. No LLM/API calls were made. No further selection was
performed using these held-out results. The first-layer algorithms and leakage
budget were unchanged; the controller overrides only its acceptance cap and source
allocation. Fresh verifier calibration is required after this policy change.

The controller now performs a bounded, configured source-only coverage search over
six tree/region/retention settings (`routing.coverage_candidates`). It selects the
inner-certified candidate with the highest benign acceptance on source tuning,
then assesses that frozen winner once on the independent outer gate. An unsafe
winner is denied; the gate is not used to try successive alternatives. An empty
acceptance policy is not marked as useful certification. This search changes
controller hyperparameters, not the standalone reference/paper algorithms, and
adds fitting time (about 116 seconds for the measured DDoS fold).

On CICIDS2018/DDoS, `finer_full_retention` was selected: tree depth 10, at most 128
leaves, four top deviations, the original region quantiles, and full source-benign
retention during membership/conditional policy selection. These filters are still
selected through the existing source policy procedure; the final gate must pass.
It accepted 2,070/5,686 source-gate benign rows, with zero leaked attack groups out
of 3,457 (exact upper bound 0.08662%). The frozen test result was 6,765/18,750 benign
flows accepted (36.08%), zero known-attack bypass out of 13,343, and zero withheld
DDoS bypass out of 5,405. Thus 30,733/37,498 test flows still escalate. The majority
benign-filtering objective remains unmet. Denser-radius candidates failed source
certification and were not promoted. This is the best of the configured source
candidates, not a proven optimum or a guarantee of unseen-attack safety. No paid
inference was run; fresh verifier calibration is required for the selected model.


The categorical verifier still returns BENIGN/ATTACK/UNCERTAIN. It uses disjoint
source tuning/certification, now independently reserved from first-layer fitting.
One randomly chosen representative per distinct input group is selected before
binary-class sampling. Up to 200 representatives/class tune the frozen categorical
policy; up to 500/class certify it. Both 1% attack-to-benign and 5% benign-to-attack
bounds must pass, using Bonferroni adjustment for joint 95% verifier confidence.
No confidence probabilities or numerical thresholds are invented. Insufficient
support, failed/incomplete calibration, or stale artifacts prohibit release.

Bounds describe source feature-group units under the sampling assumptions, not a
guaranteed error rate for repeated deployment flows, shifted distributions, or
unseen attacks. Gate and verifier checks are stage-specific, not one jointly
certified end-to-end bound. Target flow-weighted observations are reported separately.
Old calibration artifacts are invalid after the partition/code change. New artifacts
record model, source dataset, prompt/configuration, code hashes, row identities,
gate audit, source partition settings and context fingerprint. No live calibration
has been run automatically.

### Open-weight models and one investigation agent

Verification uses `openai/gpt-oss-20b`. Investigation defaults to
`qwen/qwen3-30b-a3b-instruct-2507`; `openai/gpt-oss-20b` is the paired alternative.
The [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) and
[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) model cards identify Apache-2.0
licenses. OpenRouter endpoint metadata was checked on 2026-10-23 for
[Qwen](https://openrouter.ai/api/v1/models/qwen/qwen3-30b-a3b-instruct-2507/endpoints)
and [GPT-OSS](https://openrouter.ai/api/v1/models/openai/gpt-oss-20b/endpoints):
eligible providers support tools, structured outputs and seeds. Configured price
ceilings are Qwen $0.10/$0.30 and GPT-OSS $0.018/$0.09 per million input/output tokens.
Availability and prices may change; unsupported/unavailable providers fail safely.

Both investigators share temperature zero, output/step/tool limits, prompts, tools
and permitted source references. Qwen omits reasoning settings because this is its
non-thinking instruct variant; GPT-OSS uses low reasoning. These differences and
exact profiles are saved. Seeds do not guarantee deterministic hosted inference.
Only explicitly allowed open-weight models use the existing OpenRouter HTTP client.

The single bounded agent has attack and uncertainty prompts. It reuses `inspect_flow`,
`compare_source_baseline`, and `retrieve_attack_knowledge`. Source examples contain
binary reference labels only. General catalogue knowledge is available in every
fold, including potentially withheld families: family recognition is prior-assisted,
not proof of genuinely unseen-family discovery.

Reports retain classification, family/subtype, alternative hypotheses, qualitative
evidence strength, supporting/contradicting observations, missing evidence and summary.
Every quoted observation must match an actual permitted value. Unknown/insufficient
families cannot invent a subtype; insufficient evidence may have no positive support.
Attack-mode reports cannot change classification to BENIGN. Invalid values, malformed
JSON, truncation, wrong models and tool errors are explicit. Exact-value validation
does **not** establish semantic correctness of interpretations.

Uncertainty investigation defaults off: the verifier already receives original
features and source statistics, so tools do not magically provide new packets or
host context. Optional investigation can consult general knowledge and targeted
comparisons; it must abstain when those add nothing useful. Non-abstaining
uncertainty recommendations require a successful evidence-tool consultation and
still cannot authorize automatic action.

### Commands

From the repository root, inspect support and preview requests without fitting,
API calls or saved results:

```bash
.venv/bin/python -m src.controller --dataset cicids2018 --attack DDoS --source-audit
.venv/bin/python -m src.controller --dataset cicids2018 --attack DDoS --dry-run --investigation-model both
```

Live commands below are paid operations; the program does not load `.env` itself:

```bash
set -a
source .env
set +a
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Fresh independent source calibration; never overwrite an existing artifact.
.venv/bin/python -u -m src.controller --dataset cicids2018 --attack DDoS --calibrate --output-dir results/controller/independent_v1

# Complete controller, Qwen investigation (up to 5 Layer 2 flows).
.venv/bin/python -u -m src.controller --dataset cicids2018 --attack DDoS --max-flows 5 --output-dir results/controller/independent_v1

# Paired Qwen/GPT-OSS, one frozen upstream pass on identical flow identities.
.venv/bin/python -u -m src.controller --dataset cicids2018 --attack DDoS --max-flows 50 --investigation-model both --output-dir results/controller/independent_v1

# Compare attack-only and optional uncertainty recommendations in the same run.
.venv/bin/python -u -m src.controller --dataset cicids2018 --attack DDoS --max-flows 50 --investigation-model both --investigate-uncertain --output-dir results/controller/independent_v1

# Preserve verification-only operation.
.venv/bin/python -u -m src.controller --dataset cicids2018 --attack DDoS --max-flows 5 --verify-only --output-dir results/controller/independent_v1
```

Use the same output root for calibration and evaluation. Omit `--attack` for all
configured folds, each requiring its own calibration. `--all-flows` removes the
Layer 2 limit, not the separately configured investigation sampling limit. No paid
requests or full-dataset fitting were performed during implementation.

### Paired evaluation and outputs

The controller loads data once and fits/freezes upstream decisions once per fold,
not per investigator. It then selects investigation rows by seeded hash ordering
within fold/mode/original-label strata, up to `sample_per_stratum: 5` each. Offline
labels never enter model requests. Selection support and exact IDs are saved.
Both models and all configured repeats receive the same selected identities.
Order rotates by flow; local investigation response caching is disabled. Provider
usage, including reported caching details, is retained. `repeats` defaults to one.
Budget-denied and failed records stay in the planned comparison population; an
incomplete comparison is not reported as equal successfully observed support.

Each timestamped directory under the chosen output root contains:

- `flows.csv`, `flows.jsonl`: immutable Layer 2 outcomes, flushed per flow.
- `summary.json`, `experiment_metadata.json`: source checks, routing, conditional
  verifier and full-pipeline coverage, configurations and cost.
- `investigations.csv`, `investigations.jsonl`: per-model/repeat reports, recommendation
  versus final action, permitted flow evidence, tool observations, errors, latency,
  tokens and provider usage; flushed per report.
- `investigation_summary.json`: paired identities/support, model/mode metrics, review
  rubric and first-layer-only → verified → complete-model comparison.

Binary pipeline comparisons use identical **evaluated** rows: all gate-accepted rows
plus logged Layer 2 rows, including failures. Unevaluated rows and full-test coverage
are explicit. Investigation never changes binary actions, so both complete models
have identical bypass/alert results. First-layer-only unresolved attacks are not
counted as detected. The comparison does not replay historical unsafe calibrations;
original attack-only versus optional uncertainty routing uses the same corrected,
frozen upstream population.

Family accuracy/macro-F1 use mapped malicious investigation rows, with abstentions
and errors counted as incorrect. Benign and unmapped labels are reported separately.
The fixed offline mapping is in `investigation_evaluation.label_mapping`; UNSW
Analysis is explicitly unmapped, not automatically declared an unknown family.
DoS subtypes map to their shared family, so withholding one does not make the
broader family genuinely unseen. Reported quality is conditional on the stratified
sample, not a deployment-prevalence estimate.

Automated metrics include exact-value evidence fidelity (also for rejected reports),
JSON/schema validity, tool success and mean calls, mean/P95 latency, tokens, cost per
scheduled investigation and failures. Appropriate abstention and unsupported semantic
claim rates are **unavailable until independent human review** using the configured
rubric. No second LLM is used as a ground-truth judge. Report latency covers scheduled
investigations, including failure/fallback overhead; amortized pipeline latency adds
measured first-layer prediction time per test flow and verifier time, excluding training
and offline preparation. Actual throughput and deployment latency require experiments.

The $10 application-side budget is shared across verifier/investigator calls and
resets per CLI run. Conservative cost reservations precede each attempt; there are
no retries. OpenRouter usage.cost is preferred; configured token-rate estimates are
used otherwise and missing accounting stops further paid calls. This is not a
provider-enforced spending cap. No experimental results or improvements are invented.

## Optional research comparison

The comparison runs ZETA-IDS (`zeta`), the [Fawaz et al. feature-influence reconstruction and model-uncertainty method](https://doi.org/10.1109/TNSM.2026.3731401) (`paper`), or both on the same held-out flow identities:

```bash
.venv/bin/python -m src --method zeta --dataset cicids2018 --attack 'Port Scanning'
.venv/bin/python -m src --method paper --dataset cicids2018 --attack 'Port Scanning'
.venv/bin/python -m src --method both --dataset cicids2018 --attack 'Port Scanning'
```

Omit `--attack` for every configured family; omit `--dataset` to use `comparison.datasets`. Commands without `--method` retain the terminal-only default workflow. Explicit `--method` commands use comparison settings and save reports when `comparison.save_results` is true. RF is the default paper classifier backend; XGBoost is another backend of the same method. The controller is independent of this comparison.

The paper method fits a classifier on known attacks, selects influential predicted-class TreeSHAP features, reconstructs them with an autoencoder, and combines reconstruction error with subensemble uncertainty. A validation-fitted Gaussian mixture model gives a novelty score; scores strictly above its 95th validation percentile become `ZERO_DAY_CANDIDATE`. Implementation details and parameters are in [`src/paper.py`](src/paper.py) and `config/loao.yaml`.

This is an **adapted reproduction**, not a replication of the paper's published scores. It uses this repository's datasets and group-disjoint folds instead of the paper's datasets and row splits, retains duplicates, imputes missing values, and uses one configured seed. The published 16-unit autoencoder bottleneck takes precedence over the author's 8-unit code setting. The [author implementation](https://github.com/hussein-fawaz/ZDA-SHAP-UQ) and the full list of deviations are recorded in the experiment metadata.

Comparison novelty metrics use **attack-only test flows**, with the withheld family as the positive class. ZETA-IDS `UNRESOLVED` counts as an investigation candidate for this comparison; it does not establish that a flow is malicious or genuinely new. Paper AUROC and AUPRC use its continuous novelty score; ZETA-IDS has no equivalent score. Full-traffic ZETA-IDS metrics, including benign acceptance and bypass, are reported separately.

Each explicit comparison creates a new directory under `results/comparison/` with `metrics.csv`, `summary.json`, `predictions.csv`, and `experiment_metadata.json`. The metadata records settings, data and split identities, selected features, deviations, and any failed folds. No trained models are saved.
