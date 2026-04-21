# OpenDiscoveryTrace

**Process Traces for Evaluating AI Scientist Workflows**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HF](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/aayambansall/OpenDiscoveryTrace)

---

## Overview

Existing benchmarks for AI scientific agents evaluate only final outputs. OpenDiscoveryTrace captures the **full reasoning process** — every thought, tool call, error, revision, and confidence estimate — as models work through scientific tasks.

**522 trajectories** across **7 models** (3 frontier, 4 open-weight), **124 tasks** in 4 science domains.

### Headline Result

All frontier models achieve the same success rate (~69%), yet Claude Opus 4.6 makes **30x more errors** than GPT-5.4 — predominantly tool misuse (66.7%) vs. reasoning errors (83.6%). Output-only benchmarks miss this entirely.

---

## Repository Structure

```
OpenDiscoveryTrace/
│
├── paper/                          # Submission
│   ├── paper.pdf                   #   Competition proposal (2pp + appendix)
│   ├── paper.tex                   #   LaTeX source
│   ├── references.bib              #   Bibliography
│   ├── table_comparison.tex        #   Benchmark comparison table
│   ├── figures/                    #   Figures used in paper
│   └── supplementary/              #   Full-length analysis paper (15pp)
│
├── src/
│   ├── harness/                    # Trajectory generation
│   │   ├── agent_harness.py        #   Main harness: runs models through tasks
│   │   └── run_opensource.py       #   Open-weight model runner (local GPU)
│   ├── analysis/                   # Analysis pipelines
│   │   ├── analyze_trajectories.py #   Core stats + figures
│   │   └── reviewer_analysis.py    #   Extended analysis (taxonomy, baselines)
│   └── baselines/                  # Benchmark task baselines
│       └── implement_four.py       #   IAA, sequence models, live retrieval
│
├── data/
│   ├── task_bank.json              # 200 scientific tasks (124 executed)
│   └── samples/                    # Sample trajectories (full set on HuggingFace)
│       ├── frontier/               #   3 tasks × 3 frontier models
│       └── open_weight/            #   1 task × Qwen2.5-1.5B
│
├── results/
│   ├── statistics/                 # Raw analysis outputs (JSON)
│   │   ├── analysis_results.json
│   │   ├── reviewer_analysis_results.json
│   │   └── four_additions_results.json
│   └── figures/                    # All generated figures (PDF)
│
├── research_notes/                 # Research process documentation
│   ├── literature-review.md        #   60+ papers across 5 facets
│   ├── reasoning.md                #   Hypothesis deliberation
│   ├── methodology.md              #   Pre-analysis plan
│   └── synthesis.md                #   Results interpretation
│
├── LICENSE                         # CC-BY-4.0
├── requirements.txt
└── README.md
```

---

## Quick Start

### Browse a trajectory

```python
import json

with open("data/samples/frontier/dd_e01_gpt-5.4.json") as f:
    traj = json.load(f)

print(f"Task:    {traj['prompt'][:80]}...")
print(f"Model:   {traj['model']}")
print(f"Steps:   {traj['metadata']['total_steps']}")
print(f"Errors:  {traj['metadata']['total_failures']}")
print(f"Claim:   {traj['outcome']['final_claim'][:120]}...")
```

### Run analysis

```bash
pip install -r requirements.txt
python src/analysis/analyze_trajectories.py
```

### Generate new trajectories

```bash
# Frontier models (requires API keys)
python src/harness/agent_harness.py --model gpt-5.4 --max-tasks 10

# Open-weight models (requires GPU, no API keys)
python src/harness/run_opensource.py
```

---

## Trace Schema

Each step in a trajectory records 9 fields:

| Field | Description |
|-------|-------------|
| `step_id` | Sequential step index |
| `timestamp` | ISO 8601 UTC |
| `phase` | Scientific workflow phase (literature review → hypothesis → experiment → execution → analysis → conclusion) |
| `thought` | Model's reasoning |
| `action` | Tool call details (type, tool name, input, output) |
| `observation` | Processed result of the action |
| `error` | Error state (occurred, type, message) |
| `revision_trigger` | What prompted a strategy change |
| `confidence` | Self-reported certainty \[0, 1\] |

Full JSON schema in [`paper/paper.tex`](paper/paper.tex) Appendix A.

---

## Models

| Model | Type | Trajectories | Tasks |
|-------|------|-------------|-------|
| GPT-5.4 | Frontier | 124 | 124 |
| Claude Opus 4.6 | Frontier | 124 | 124 |
| Gemini 3.1 Pro | Frontier | 124 | 124 |
| Qwen2.5-7B-Instruct | Open-weight | 30 | 30 |
| Mistral-7B-v0.3 | Open-weight | 30 | 30 |
| Phi-3.5-mini-instruct | Open-weight | 30 | 30 |
| Qwen2.5-1.5B-Instruct | Open-weight | 30 | 30 |

---

## Benchmark Tasks

1. **Trajectory Outcome Prediction** — predict success from step features
2. **Error Localization** — identify the step where reasoning went wrong
3. **Claim Verification** — verify correctness of final claims
4. **Autonomy Classification** — classify L1–L4 autonomy levels
5. **Process Quality Scoring** — multi-axis trajectory quality

Baselines in [`results/statistics/`](results/statistics/) and Appendix H of the paper.

---

## Full Dataset

Sample trajectories are included in `data/samples/`. The complete 522-trajectory dataset is on HuggingFace:

**[huggingface.co/datasets/aayambansall/OpenDiscoveryTrace](https://huggingface.co/datasets/aayambansall/OpenDiscoveryTrace)**

---

## Citation

```bibtex
@inproceedings{opendiscoverytrace2026,
  title     = {OpenDiscoveryTrace: Process Traces for Evaluating
               AI Scientist Workflows},
  author    = {Anonymous},
  booktitle = {AI for Science Workshop, ICML},
  year      = {2026},
  note      = {Dataset Proposal Competition}
}
```

## License

Code: MIT. Data and paper: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
