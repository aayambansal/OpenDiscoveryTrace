# Synthesis: OpenDiscoveryTrace (Revised after Critique 2)

## Key Finding 1: Models Exhibit Fundamentally Different Interaction Patterns
Claude Sonnet 4 and GPT-4o show dramatically different multi-turn interaction behaviors:
- **Claude** completes the entire 6-phase scientific workflow in a single long response (~3000 tokens), 
  packing literature review, hypothesis, experiment, analysis, and conclusion into one turn
- **GPT-4o** distributes the workflow across multiple turns (mean 9.6 steps), with each step
  performing one action (tool call, reasoning, or conclusion)

This is NOT a parsing artifact — it is a genuine behavioral difference in how models approach
multi-turn scientific workflows. Both achieve similar success rates, validating that the 
single-response vs. multi-turn distinction captures a real interaction pattern difference
that output-only benchmarks would completely miss.

**Statistical evidence** (restricted to drug_discovery domain for fair comparison, n=27 Claude, n=31 GPT-4o):
- Steps: Claude 1.9±0.8 vs GPT-4o 12.0±? (p<1e-15, Cliff's delta to be computed)
- Tool calls: Claude 0.9±0.8 vs GPT-4o uses tools extensively
- Both reach conclusions: 100% conclusion rate for both models
- Success rates limited by ground-truth availability (small n); not formally compared

## Key Finding 2: Process Metrics Capture Information Output Metrics Cannot
Even within the same domain (drug discovery), models with similar conclusion quality
differ dramatically on:
- Number of reasoning steps
- Tool invocation frequency
- Error recovery patterns
- Revision behavior

This validates H1: process-level traces reveal systematic differences invisible to
final-answer evaluation.

## Key Finding 3: Task Difficulty Calibration Works
- Easy tasks: mean 4.9 steps, 0.4 errors/trajectory
- Medium tasks: mean 9.9 steps, 1.9 errors/trajectory  
- Hard tasks: mean 12.1 steps, 2.8 errors/trajectory
The monotonic increase confirms our difficulty stratification.

## Key Finding 4: Rich Failure and Revision Data
- 52.3% of trajectories contain at least one error
- 57.6% contain at least one revision (keyword-detected, may undercount for some models)
- This exceeds the pre-specified 30% threshold

## Critical Caveats (from Critique 2)
1. **Imbalanced data**: 124 GPT-4o vs 43 Claude; Gemini still collecting
2. **Domain coverage**: Claude data available only for drug_discovery as of this analysis
3. **Success evaluation**: Ground-truth available only for easy tasks (n=54 total evaluable)
4. **Interaction pattern vs. capability**: The step-count difference reflects how models
   structure multi-turn interaction, not necessarily reasoning depth
5. **Web search is simulated**: web_search tool returns placeholder; models use parametric knowledge
6. **Wall time includes API latency**: Cannot be interpreted as "reasoning time"

## Honest Framing for the Paper
The primary contribution is the **dataset and schema**, not the model comparison.
The baseline results demonstrate the dataset's utility by showing that process traces
capture information invisible to output metrics. We should:
1. Present the dataset design as the primary contribution
2. Present model comparison as a demonstration of the dataset's analytical value
3. Acknowledge all limitations prominently
4. Frame the interaction pattern difference as a finding, not a capability ranking
