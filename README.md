# OpenDiscoveryTrace

**Process Traces for Evaluating AI Scientist Workflows**

*Submitted to the AI for Science Workshop Dataset Competition (ICML 2026)*

---

OpenDiscoveryTrace is a dataset of **372 complete AI scientific agent trajectories** capturing how frontier language models reason through scientific tasks — not just what they produce.

Each trajectory records a **9-field-per-step trace** (thoughts, tool calls, observations, errors, revisions, confidence) as models execute tasks across drug discovery, materials science, genomics, and scientific literature analysis.

## Key Finding

All three frontier models achieve indistinguishable success rates (~69%), yet **Claude Opus 4.6 produces 30x more errors than GPT-5.4** (2.5 vs 0.08 per trajectory, *p* < 0.0001, Cliff's δ = 0.613) while arriving at the same answers. These errors are qualitatively different: Claude's are 66.7% tool misuse; GPT-5.4's are 83.6% reasoning errors.

**Process traces expose this. Output-only benchmarks cannot.**

## Dataset Summary

| | |
|---|---|
| **Trajectories** | 372 (124 per model, balanced) |
| **Models** | GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro |
| **Open-source** | + 30 Qwen2.5-1.5B trajectories |
| **Tasks** | 124 executed across 4 domains × 3 difficulties |
| **Domains** | Drug Discovery, Materials Science, Genomics, Literature |
| **Schema** | 9 fields per step (ReAct-extended) |
| **Benchmark Tasks** | 5 defined with baselines |
| **License** | CC-BY-4.0 |

## Repository Structure

```
OpenDiscoveryTrace/
├── README.md
├── LICENSE                          # CC-BY-4.0
├── requirements.txt
│
├── data/
│   ├── task_bank.json               # 200 scientific tasks (124 executed)
│   ├── trajectories_sample/         # 9 sample trajectories (3 tasks × 3 models)
│   └── trajectories_opensource_sample/  # Qwen2.5-1.5B sample
│
├── src/
│   ├── agent_harness.py             # Trajectory generation pipeline
│   ├── analyze_trajectories.py      # Analysis and figure generation
│   ├── reviewer_analysis.py         # Extended analysis (baselines, taxonomy, IAA)
│   ├── implement_four.py            # IAA, sequence baselines, open-source, live retrieval
│   └── run_opensource.py            # Open-source model trajectory generation
│
├── analysis/
│   ├── analysis_results.json        # Core dataset statistics
│   ├── reviewer_analysis_results.json  # Extended analysis results
│   └── four_additions_results.json  # IAA, sequence baselines, live retrieval results
│
├── figures/                         # Publication-quality figures (PDF)
│
├── paper/
│   ├── paper.tex                    # 2-page dataset proposal (competition version)
│   ├── paper.pdf                    # Compiled PDF
│   ├── references.bib
│   ├── table_comparison.tex
│   ├── figures/                     # Paper figure copies
│   └── full_version/               # Full 15-page analysis paper (archived)
│
├── literature-review.md             # 60+ papers across 5 facets
├── reasoning.md                     # Research design deliberation
├── methodology.md                   # Pre-analysis plan
└── synthesis.md                     # Results interpretation
```

## Quick Start

### View sample trajectories
```python
import json
with open("data/trajectories_sample/dd_e01_gpt-5.4.json") as f:
    traj = json.load(f)
print(f"Task: {traj['prompt'][:100]}")
print(f"Steps: {traj['metadata']['total_steps']}")
print(f"Errors: {traj['metadata']['total_failures']}")
print(f"Claim: {traj['outcome']['final_claim'][:200]}")
```

### Run analysis on sample data
```bash
pip install -r requirements.txt
cd src && python analyze_trajectories.py
```

### Generate new trajectories
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
python src/agent_harness.py --model gpt-5.4 --max-tasks 10 --concurrency 2
```

## Trace Schema

Each step records 9 fields:

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | int | Step index |
| `timestamp` | ISO 8601 | When the step occurred |
| `phase` | enum | literature_review, hypothesis, experiment_design, execution, analysis, revision, conclusion |
| `thought` | string | Agent's reasoning |
| `action` | object | Tool call (type, tool, input, output) |
| `observation` | string | Result of the action |
| `error` | object | Error state (occurred, type, message) |
| `revision_trigger` | string | What prompted a strategy change |
| `confidence` | float | Agent's self-reported certainty [0, 1] |

## Benchmark Tasks

1. **Trajectory Outcome Prediction** — Predict success from early steps
2. **Error Localization** — Identify where reasoning went wrong
3. **Claim Verification** — Verify correctness of final claims
4. **Autonomy Level Classification** — Classify L1-L4 autonomy
5. **Process Quality Scoring** — Multi-axis trajectory quality

## Full Dataset

The complete 372-trajectory dataset is available on HuggingFace:

**[huggingface.co/datasets/aayambansall/OpenDiscoveryTrace](https://huggingface.co/datasets/aayambansall/OpenDiscoveryTrace)** 

## Citation

```bibtex
@inproceedings{opendiscoverytrace2026,
  title     = {OpenDiscoveryTrace: Process Traces for Evaluating AI Scientist Workflows},
  author    = {Anonymous},
  booktitle = {AI for Science Workshop, ICML},
  year      = {2026},
  note      = {Dataset Proposal Competition}
}
```

## License

Dataset and code released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
