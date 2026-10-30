# halluc-detect

> Agent trajectory hallucination detector — the open-source Phase 1 flagship of [HallucC](https://aihcc.cloud).
> **The fast tier is pure Python stdlib: 0 LLM calls / 0 web searches, fully offline-reproducible.**

**[English](README.md) | [中文版](README.zh-CN.md)**

---

## What it is

`halluc-detect` is the **Phase 1 open-source flagship** of [HallucC](https://aihcc.cloud): a hallucination detector for LLM **agent execution trajectories** (not just final text), plus a Chinese regression benchmark. It turns "will the agent fabricate results with misplaced confidence" into something **runnable, reproducible, and zero-cost**.

- **Detector**: runs tiered evaluation over an agent trajectory (thought / action / action_input / observation / agent_claim sequence) and outputs **six-dimension scores + per-step checks + 12 failure-mode codes + hallucination propagation chains**.
- **Benchmark**: 100 Chinese agent trajectories (HallucC-Agent-Bench) = 20 contract regression cases (planted known hallucinations, recall=1.0 regression gate) + 80 adversarial cases (real distribution, quantifying generalization limits). Fast tier: 0 LLM / 0 search, real recall=0.945 / fp_rate=0.0.
- **Independent circularity probe**: an independent LLM-as-judge (DeepSeek temp=0, blind — sees neither author labels nor detector output) re-labels all 100 cases, confirming ~86% of hallucination labels + 100% of clean ones (specificity=1.0, zero false positives), with a **low circularity-risk** verdict — breaking the "case author = detector author = annotator" circularity concern.
- **Bonus dataset**: `eval_set/chinese_hallucination_benchmark.jsonl` — 100 annotated Chinese hallucination samples (~260 claims, 9 hallucination categories: fabricated facts, misattribution, data errors, temporal errors, etc.), CC-BY-4.0. See `eval_set/README.md`.

> **Positioning**: this repo is a **detector + regression benchmark**, not an academic novelty claim.

## Why the fast tier has zero dependencies

The fast tier runs only the rule layer: regex + token similarity + arithmetic verification + common-sense contradictions + a bilingual marker lexicon. The rule layer was deliberately decoupled — `import agent_pipeline` no longer drags in the `server`/`config`/`llm`/`search` heavy stack, so:

```bash
pip install halluc-detect          # no extras needed for the fast tier
halluc-detect bench                # 0 API quota, offline reproducible
```

The `standard` / `deep` tiers require the LLM judge layer (see [optional dependencies](#install)) and correspond to HallucC's paid SaaS.

## Install

```bash
# Dev mode (inside the repo)
git clone <repo> && cd HallucC-Agent-Bench
pip install -e .

# Fast tier only (0 dependencies)
pip install halluc-detect

# LLM judge layer (standard / deep tiers)
pip install "halluc-detect[llm]"

# NLI semantic entailment (observation↔claim contradiction detection)
pip install "halluc-detect[nli]"
```

> **Editable install**: run `pip install -e .` at the repo root. The wheel packages 13 open-source modules (including the slimmed `llm.py` client — see `py-modules` in `pyproject.toml`). **Open-core boundary**: production judge prompts (`prompts.py`) and server-side modules such as `pipeline`/`config`/`search`/`domains` are **not** in this repo — `import agent_pipeline` does not pull them in; the fast tier is dependency-free; the standard/deep judge layer runs on the SaaS side.

## Usage

### `detect` — check a single trajectory

```bash
# Bare trajectory list via stdin (no tool_schemas → HALLUCINATED_TOOL/SCHEMA_VIOLATION skipped)
echo '[
  {"step":1,"action":"search","action_input":"{\"query\":\"order 12345\"}",
   "observation":"order not found","agent_claim":"I found the details of order 12345"}
]' | halluc-detect detect

# File input (may include task / tool_schemas)
halluc-detect detect trajectory.json --speed fast

# Enable tool-schema validation (loads the benchmark's 5-tool registry)
halluc-detect detect trajectory.json --tools demo
```

Each step accepts two schemas (normalized by `agent_eval._normalize_step`):

| schema | fields |
|---|---|
| A | `step, thought, action, action_input, observation, final_answer` |
| B | `step, tool, tool_input, tool_output, agent_claim` |

Output is structured JSON (`evaluate_trajectory().to_dict()`): six-dimension `dimensions`, per-step `steps[].checks`, the `failure_modes` list, the `propagation` chain, and `llm_calls`/`search_calls` (both 0 in the fast tier).

### `bench` — run the Chinese benchmark

```bash
# Text report (fast tier by default)
halluc-detect bench

# Structured JSON
halluc-detect bench --json

# Standard tier (requires [llm] extra + API key)
halluc-detect bench --speed standard
```

Fast-tier acceptance gates: `recall ≥ 0.80`, `fp_rate ≤ 0.15`. Current fast tier (100 cases):

| suite | n | recall | fp_rate | notes |
|---|---|---|---|---|
| contract regression | 20 | 1.0 | 0.0 | planted known hallucinations, regression gate (by design) |
| adversarial | 80 | 0.937 | 0.0 | real distribution; all 5 FNs are COVERAGE_GAP (LLM-only modes, fast tier by design does not cover) |
| total | 100 | 0.945 | 0.0 | 0 LLM / 0 search |

Persisted snapshot: `eval_set/agent_benchmark_results.json` (includes a standard-tier subset of 20, recall=1.0).

### `judge` — independent circularity probe

```bash
# Independent LLM judge blind-labels all 100 cases (requires [llm] extra + API key)
python agent_benchmark_judge.py
# Sampled smoke run
python agent_benchmark_judge.py --limit 20
```

The judge sees only `{task, trajectory}` + the tool registry — not the author's `expected_failure_modes`, not the detector's output — and independently decides whether each step is grounded. It produces three-way agreement rates (judge↔author labels, judge↔author codes, judge↔detector) + template-effective-evidence deduplication + a circularity-risk verdict.

Current snapshot (`eval_set/judge_relabel.json`, DeepSeek temp=0): judge recall=0.857 / specificity=1.0 (TP=78 FN=13 TN=9 FP=0), template redundancy reduced 100→41 independent pieces of evidence, circularity-risk verdict **low**. DeepSeek temp=0 is not fully deterministic; a single run carries ±1 FN noise.

## Failure-mode codes (12)

**Deterministic (reportable in fast tier, 10):**

| code | meaning |
|---|---|
| `HALLUCINATED_TOOL` | calling a tool not in the registry |
| `PARAM_FABRICATION` | parameter contains a rigid ID (order number / UUID) with no source in context |
| `SCHEMA_VIOLATION` | parameters violate the tool's JSON Schema |
| `OBSERVATION_IGNORED` | observation explicitly failed / is empty, yet success is claimed |
| `CLAIM_MISMATCH` | observation and claim disagree (similarity below threshold / arithmetic wrong / NLI contradiction) |
| `LOOP` | same action+input repeated |
| `RETRY_STORM` | same tool with ≥3 different inputs including failed observations |
| `REDUNDANT_STEP` | different calls but highly duplicated observations (≥0.8) |
| `HANDOFF_LOSS` | a later claim corrects / overturns an earlier claim |
| `OVERCONFIDENT_ASSERTION` | common-sense contradiction (impossible date / percentage >100 / year paradox) |

**LLM-only (standard / deep tiers, 2):** `SCOPE_DRIFT` (drifting off-task mid-run), `PREMATURE_STOP` (final answer fails to deliver the task).

## Differentiation

- **Bilingual marker lexicon**: CN literals preserved verbatim + EN word-boundary matching (`\b` avoids false hits like "foundry"/"nullable") + JSON/HTTP-scoped regex (`"status": 503` avoids misreading bare numbers as arithmetic results).
- **CN-first multi-domain benchmark**: 100 trajectories covering orders / search / calculation / translation / weather / medical / finance / logistics (20 contract + 80 adversarial) — existing agent-hallucination benchmarks skew EN.
- **Circularity self-evidence**: the blind independent judge breaks the "detector catches its own templates" concern — if that were true, recall would approach 0, not the measured 0.857.
- **Tiered evaluation + false-positive control**: insufficient evidence → `unverified`, never forced `refuted` (consistent with the HallucC main line). The rule layer hands only genuinely "uncertain" items to the judge layer, avoiding full-volume LLM calls.

**Phase 2 (planned)**: public leaderboard (detector axis + agent axis).

## License

- **Code**: MIT (see `LICENSE`).
- **Benchmark datasets** (HallucC-Agent-Bench 100 trajectory cases + expected labels; Chinese hallucination eval set): CC-BY-4.0.

> Two-tier licensing: MIT for code encourages citation and derivative work; CC-BY-4.0 for datasets keeps the attribution requirement.
