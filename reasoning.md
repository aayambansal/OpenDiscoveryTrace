# Research Deliberation: OpenDiscoveryTrace

## Knowledge Consolidation

The literature reveals a clear gap at the intersection of three mature research areas. (1) **AI science benchmarks** (ScienceAgentBench, DiscoveryBench, PaperBench) evaluate final outputs but not the discovery process. (2) **Agent trajectory datasets** (WebArena, ToolBench, OpenHands) record rich process traces but only for web/software domains. (3) **Autonomous science systems** (Coscientist, Robin, AI Scientist) execute the full scientific loop but publish no reusable trace data. The workshop call explicitly states: "Datasets that capture the full scientific stack—from planning prompts to robotic execution traces—are urgently needed."

**What is established:**
- The ReAct (Thought-Action-Observation) trace format is the de facto standard for agent trajectories
- Datasheets for Datasets (Gebru et al.) is the expected documentation standard
- L0-L5 autonomy levels are proposed but unvalidated taxonomies
- Existing benchmarks are product-focused, not process-focused

**What is contested:**
- Whether AI systems can meaningfully recover from scientific failures (some evidence from MINT showing feedback helps, but not in scientific domains)
- Whether computational-only experiments are sufficient for evaluating scientific agency (Robin argues wet-lab is essential; AI Scientist operates purely computationally)

**What is unknown:**
- How frontier models differ in their scientific reasoning trajectories (no comparative data exists)
- What fraction of AI scientific trajectories involve meaningful failure recovery vs. simple retries
- Whether human intervention points are predictable from trace features

## Knowledge Gaps & Contradictions

- **Gap 1**: No structured trace dataset for AI scientific discovery exists at all
- **Gap 2**: No failure taxonomy for AI scientific agents (unlike software bugs which are well-categorized)
- **Gap 3**: No process-level evaluation metrics for scientific trajectories
- **Gap 4**: Autonomy taxonomies (L0-L5) have no empirical validation against real traces
- **Contradiction**: AI Scientist v2 was accepted at an ICLR workshop, suggesting computational-only science can pass peer review; yet Robin and Coscientist argue that physical validation is essential for real discovery. Our dataset can include both computational and physical task designs.

## Candidate Hypotheses

### Hypothesis 1: Full process traces reveal systematic differences between frontier models that final-output benchmarks miss
- **Null hypothesis**: Model rankings on trajectory-level metrics are identical to rankings on final-output metrics
- **Required evidence**: Run 3 models on same tasks, compare rankings on output metrics vs. process metrics
- **Feasibility**: HIGH — we control the task set and can compute both metric types
- **Novelty**: HIGH — no prior comparative study of scientific process traces exists
- **If confirmed**: Process metrics provide complementary signal for model evaluation
- **If refuted**: Final-output benchmarks are sufficient (also interesting, simplifies evaluation)

### Hypothesis 2: Failed trajectories contain predictable patterns that distinguish recoverable from terminal failures
- **Null hypothesis**: Failure patterns are random with respect to recovery success
- **Required evidence**: Annotate failure types and recovery outcomes; test association
- **Feasibility**: MEDIUM — requires enough failed trajectories with attempted recovery
- **Novelty**: HIGH — no failure taxonomy for AI scientific agents exists
- **If confirmed**: Enables design of better failure recovery mechanisms
- **If refuted**: Failures may be too diverse to categorize (still informative for the community)

### Hypothesis 3: Human intervention points can be predicted from trace features before the agent fails
- **Null hypothesis**: Trace features before failure are indistinguishable from normal operation
- **Required evidence**: Train classifiers on trace features to predict upcoming human intervention need
- **Feasibility**: LOW for this paper — requires substantial annotation of intervention points
- **Novelty**: HIGH — connects to governance and oversight
- **If confirmed**: Enables proactive human oversight systems
- **If refuted**: Human oversight must remain continuous (also important finding)

## Structured Deliberation

| Hypothesis | Strengths | Weaknesses | Key Uncertainty | Information Gain |
|------------|-----------|------------|-----------------|-----------------|
| H1: Process traces reveal model differences | Directly testable with our data; compelling for benchmark contribution | May require large N for statistical power | Whether differences are practically significant | HIGH — foundational for the dataset's value proposition |
| H2: Failure patterns are predictable | Unique contribution; actionable for system design | Requires enough failures; taxonomy design is subjective | Whether our failure taxonomy is comprehensive | MEDIUM — secondary contribution |
| H3: Intervention prediction | Most forward-looking; governance-relevant | Requires human annotation of intervention points which is expensive | Whether agents give warning signs before failure | LOW — exploratory, better for future work |

## Selected Direction

- **Primary focus**: The dataset itself as the contribution (schema + traces + benchmark tasks)
- **Supported hypothesis**: H1 — process traces reveal information that final-output benchmarks miss
- **Secondary analysis**: H2 — failure pattern characterization (descriptive, not predictive)
- **Deferred**: H3 — intervention prediction (position as future work)

**Rationale**: This is a dataset paper for the Dataset Proposal Competition. The primary contribution is the artifact (schema, data, benchmarks), not a single hypothesis. H1 validates the dataset's utility. H2 provides additional analytical depth. H3 is future work motivation.

**Key risks**:
1. API costs exceed budget → mitigate with task count control (start with 50 tasks per domain = 200 total, 600 trajectories across 3 models)
2. Models refuse scientific tasks → document refusals as data points
3. Trajectories are too similar across models → would still be useful as a baseline dataset
4. Time pressure (13 days) → parallelize generation across models; use Lambda cluster for orchestration

**Pre-specified success criteria**:
1. 600+ trajectories collected (3 models × 200 tasks)
2. At least 30% of trajectories contain meaningful failures
3. At least 5 benchmark tasks defined with baseline scores
4. Statistically significant differences between at least 2 models on at least 1 process metric
5. Public release on HuggingFace

**Fallback plan**: If API costs or time prevent full 600 trajectories, release a smaller but well-documented pilot dataset of 150 trajectories (3 models × 50 tasks) and frame as a pilot study with expansion plans.

## Design Decisions

### Domain Selection (4 domains)
1. **Drug Discovery** — ADMET prediction, target identification, compound design. Rich tool ecosystem (RDKit, PubChem). Ground-truth available from TDC.
2. **Materials Science** — Property prediction, stability analysis, synthesis planning. Ground-truth from Materials Project. Links to A-Lab narrative.
3. **Genomics/Biology** — Gene function, pathway analysis, variant interpretation. Rich databases (UniProt, KEGG). Links to Robin narrative.
4. **Scientific Literature Analysis** — Systematic review, claim verification, contradiction detection. Links to PaperQA2. Ground-truth from expert evaluations.

### Task Difficulty Spectrum
- **Easy** (25%): Single-tool, well-defined tasks (e.g., "Predict the LogP of aspirin")
- **Medium** (50%): Multi-step, requires tool chaining (e.g., "Find all kinase inhibitors with IC50 < 100nM for target X and predict their ADMET profiles")
- **Hard** (25%): Open-ended, requires hypothesis formulation (e.g., "Propose a novel mechanism for drug resistance in cancer cell line Y based on available genomic data")

### Trace Schema Design
Extend ReAct format with science-specific fields:
```json
{
  "task_id": "str",
  "domain": "drug_discovery|materials|genomics|literature",
  "difficulty": "easy|medium|hard",
  "prompt": "str",
  "ground_truth": "str|null",
  "model": "gpt-4o|claude-sonnet|gemini-pro",
  "trajectory": [
    {
      "step_id": int,
      "timestamp": "ISO8601",
      "phase": "literature_review|hypothesis|experiment_design|execution|analysis|revision|conclusion",
      "thought": "str",
      "action": {"type": "str", "tool": "str", "input": {}, "output": {}},
      "observation": "str",
      "error": {"occurred": bool, "type": "str", "message": "str"},
      "revision_trigger": "str|null",
      "confidence": float
    }
  ],
  "outcome": {
    "success": bool,
    "final_claim": "str",
    "verification": {"method": "str", "result": "str", "score": float},
    "failure_type": "str|null",
    "recovery_attempted": bool,
    "recovery_successful": bool
  },
  "metadata": {
    "total_steps": int,
    "total_tokens": int,
    "total_tool_calls": int,
    "total_failures": int,
    "total_revisions": int,
    "wall_time_seconds": float,
    "autonomy_level": "L1|L2|L3|L4"
  }
}
```

### Failure Taxonomy (preliminary)
1. **Hallucination**: Agent fabricates data, citations, or tool outputs
2. **Tool misuse**: Incorrect API call, wrong parameters, misinterpretation of results
3. **Reasoning error**: Logical fallacy, incorrect inference, flawed statistical reasoning
4. **Knowledge gap**: Agent lacks domain knowledge to proceed
5. **Scope creep**: Agent diverges from the task into unrelated territory
6. **Premature conclusion**: Agent concludes before sufficient evidence
7. **Circular reasoning**: Agent repeats same failed approach without adaptation
8. **Refusal**: Agent refuses to attempt the task

### Benchmark Tasks (5 tasks)
1. **Trajectory Outcome Prediction**: Given first k steps, predict success/failure
2. **Error Localization**: Given a failed trajectory, identify the step where reasoning went wrong
3. **Claim Verification**: Given the agent's final claim and trajectory, verify correctness
4. **Autonomy Level Classification**: Classify the trajectory into L1-L4 autonomy levels
5. **Process Quality Scoring**: Rate overall trajectory quality on 5 axes (correctness, efficiency, tool use, recovery, calibration)
