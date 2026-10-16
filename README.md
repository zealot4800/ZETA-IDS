# ZETA-IDS

`config/system.yaml` sets the dataset, frozen artifacts, feature schema, sampling,
routing, memory, and replay. `config/agent.yaml` sets prompts, tools, verdict rules,
and budgets. Local `.env` supplies `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and
`OPENROUTER_BASE_URL`; `MAX_LLM_CALLS` and `MAX_OUTPUT_TOKENS` override the matching
agent limits when present. Mock mode is the default. Live requests require
`--backend openrouter --allow-paid`.

```bash
.venv/bin/python -m src simulate --dataset cic2017
```

Memory state and optional independent reviews use YAML. JSON is used only by the
OpenRouter API protocol and the unchanged frozen feature fingerprint.
