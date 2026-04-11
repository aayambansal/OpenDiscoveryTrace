# Methodology: OpenDiscoveryTrace

## Research Question & Hypothesis
- **Question**: Can a standardized trace dataset of AI scientific agent trajectories reveal systematic differences between frontier models that final-output benchmarks miss?
- **Hypothesis (H1)**: Process-level metrics (failure recovery rate, revision frequency, tool use efficiency) provide ranking signals that differ from output-only metrics.
- **Success criteria**: (a) 600+ trajectories collected; (b) ≥30% contain failures; (c) statistically significant inter-model differences on ≥1 process metric (p<0.05, Bonferroni-corrected).

## Data Sources
- **Task bank**: 200 scientific tasks designed by us, stratified across 4 domains × 3 difficulty levels
  - Drug Discovery (50 tasks): sourced from TDC benchmarks, PubChem queries, ADMET prediction
  - Materials Science (50 tasks): sourced from Materials Project, stability prediction, synthesis planning  
  - Genomics/Biology (50 tasks): sourced from UniProt, KEGG, gene function analysis
  - Scientific Literature (50 tasks): systematic review, claim verification, contradiction detection
- **Models**: GPT-4o (OpenAI), Claude Sonnet 4 (Anthropic), Gemini 2.5 Pro (Google)
- **Tools provided to agents**: Python execution, web search, PubMed API, PubChem API, UniProt API, RDKit, Materials Project API

## Analysis Pipeline

### Step 1: Task Bank Design
- Method: Manually design 200 tasks with ground-truth answers where possible
- Difficulty distribution: 25% easy, 50% medium, 25% hard
- Each task has: prompt, domain label, difficulty label, ground-truth (if available), required tools list
- Validation: 3 tasks pilot-tested per domain to calibrate difficulty

### Step 2: Agent Harness Construction
- Method: Build a standardized Python harness that:
  1. Presents each task to the model via its API
  2. Provides a common tool suite (Python REPL, web search, domain APIs)
  3. Logs every step as structured JSON following our trace schema
  4. Implements a multi-phase scientific workflow prompt:
     Phase 1: Literature Review → Phase 2: Hypothesis → Phase 3: Experiment Design → Phase 4: Execution → Phase 5: Analysis → Phase 6: Conclusion
  5. Allows up to 30 steps per trajectory (prevents infinite loops)
  6. Records errors, retries, and tool call failures
- Tools: Python, OpenAI/Anthropic/Google APIs, subprocess for code execution
- Output: 600 trajectory JSON files (3 models × 200 tasks)

### Step 3: Trajectory Generation (COMPUTE)
- Platform: Lambda Labs 8×V100 cluster (orchestration only; API calls to frontier models)
- Parallel execution: 8 concurrent trajectories per model
- Estimated time: ~3-4 days continuous
- Estimated API cost: ~$100-200 total across 3 providers
- Checkpointing: Save each trajectory immediately upon completion

### Step 4: Automated Verification [FIX: Blocking Issue #1 — Cross-Judge Design]
- Method: For tasks with ground-truth, automatically verify final claims via exact match / numerical tolerance
- For open-ended tasks: **Cross-model judging** — each model's trajectories are judged by the OTHER two models
  - GPT-4o trajectories judged by Claude Sonnet 4 and Gemini 2.5 Pro
  - Claude trajectories judged by GPT-4o and Gemini 2.5 Pro
  - Gemini trajectories judged by GPT-4o and Claude Sonnet 4
  - Report inter-judge agreement (Cohen's kappa between two judges per trajectory)
  - Final score = average of two cross-model judge scores
- Metrics: Binary success/failure, claim quality score (1-5), factual accuracy
- This eliminates self-evaluation bias where a model rates its own outputs

### Step 5: Human Annotation [FIX: Blocking Issues #2, #4, #5]
- Method: Authors + 1 external domain-expert annotator annotate a stratified sample of 120 trajectories
- **Calibration session**: All 3 annotators discuss and align on 10 example trajectories before independent annotation
- **Overlap**: 60 trajectories annotated by all 3 annotators (50% overlap for stable reliability estimates)
- Annotation axes:
  1. Correctness of reasoning (1-5 ordinal)
  2. Tool use efficiency (1-5 ordinal)
  3. Recovery quality after failure (1-5 ordinal, or N/A if no failure)
  4. Autonomy level classification (L1-L4) — using operational definitions with worked examples (see Appendix)
  5. Failure type (from our 8-type taxonomy, with "Other" option) if applicable
  6. **Error step identification** (for failed trajectories): annotators mark the first step where reasoning went wrong [FIX #4]
- Inter-annotator agreement: 
  - **Weighted kappa** (quadratic weights) for ordinal axes (1-5 scales) — reported per axis
  - **Fleiss' kappa** for 3 annotators on nominal axes (failure type, autonomy level)
  - Required: weighted kappa ≥ 0.6 per axis on calibration subset before full annotation
- **Autonomy Classification** [FIX #5]: Presented as a *proposed* taxonomy with operational definitions:
  - L1: Agent follows explicit instructions, minimal autonomous decisions
  - L2: Agent selects tools and methods autonomously within a specified domain
  - L3: Agent formulates sub-hypotheses and adapts strategy based on intermediate results
  - L4: Agent proposes novel research directions and designs experiments without explicit guidance
  - If inter-annotator kappa < 0.6 on autonomy levels, downgrade to exploratory analysis

### Step 6: Baseline Benchmarks
- Run 5 benchmark tasks:
  1. **Trajectory Outcome Prediction**: Random forest on step-level features → predict success/failure
  2. **Error Localization**: Human-labeled error steps (from annotation Step 5.6) as ground-truth; evaluate heuristic baselines (last tool error, confidence drop) and LLM-based localization on the annotated subset of failed trajectories. Report with bootstrap 95% CIs given smaller evaluation set.
  3. **Claim Verification**: Ground-truth comparison for easy/medium tasks
  4. **Autonomy Classification**: Proposed taxonomy (L1-L4) with operational definitions; report inter-annotator agreement as primary metric. If kappa ≥ 0.6, evaluate LLM-based classifier against human labels. If kappa < 0.6, report as exploratory analysis only.
  5. **Process Quality Scoring**: Correlate automated metrics with human annotations

### Step 7: Statistical Analysis
- Inter-model comparison: Kruskal-Wallis test across 3 models on each metric
- Post-hoc: Dunn's test with Bonferroni correction for pairwise comparisons
- Effect sizes: Cliff's delta for non-parametric comparisons
- Domain × Model interaction: Two-way analysis on success rates

## Controls & Validation
- **Positive control**: Include 10 "trivial" tasks per domain where all models should succeed (validates harness works)
- **Negative control**: Include 5 "impossible" tasks per domain where no model should succeed (validates failure detection)
- **Reproducibility**: Fix random seeds, record all API parameters, use temperature=0 where possible
- **Annotation quality**: Weighted kappa ≥ 0.6 per axis on 60-trajectory overlap required before analysis proceeds

## Statistical Plan [FIX: Blocking Issue #3 — Two-Stage Testing Hierarchy]
- **Stage 1 (Omnibus)**: 4 Kruskal-Wallis H-tests (one per primary metric: success rate, mean step count, failure recovery rate, tool use efficiency)
  - Holm-Bonferroni correction across 4 omnibus tests (uniformly more powerful than plain Bonferroni)
  - Family-wise α = 0.05
- **Stage 2 (Post-hoc)**: Dunn's test only for metrics where Stage 1 is significant
  - 3 pairwise comparisons per significant metric (GPT-4o vs Claude, GPT-4o vs Gemini, Claude vs Gemini)
  - Holm-Bonferroni correction within each metric family
- **Effect size**: Cliff's delta (small: |d|>0.147, medium: |d|>0.33, large: |d|>0.474)
- **Domain × Model interaction**: Logistic regression with domain and model as factors for binary success outcome (more appropriate than ANOVA on rates)
- **Pre-analysis plan**: All tests specified before data collection; deviations documented

## Compute Requirements
- **Platform**: Lambda Labs 8×V100 (already provisioned)
- **GPU usage**: Orchestration only (no model inference on GPUs; inference via API)
- **Estimated duration**: 3-4 days for trajectory generation
- **Estimated API cost**: ~$300-600 total (revised upward: 600 trajectories × ~15 steps × ~2K tokens ≈ 18M tokens across 3 providers)
- **Storage**: ~2-5 GB for 600 trajectory JSON files

## Limitations & Assumptions
- **Assumption 1**: Frontier model APIs are stable during collection (mitigate: record model versions)
- **Assumption 2**: Our tool suite is representative of what AI scientists need (mitigate: extensible schema)
- **Limitation 1**: Computational tasks only; no physical lab execution
- **Limitation 2**: 200 tasks may not cover all scientific reasoning patterns
- **Limitation 3**: LLM-as-judge verification has known biases (mitigate: use only for open-ended tasks, report agreement with human annotations)
- **Limitation 4**: Temperature=0 may not capture full model capability distribution
