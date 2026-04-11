# Literature Review: OpenDiscoveryTrace

## Summary

Our review across 5 facets (60+ papers) reveals a striking structural gap in the AI-for-science ecosystem: **no existing dataset captures the full end-to-end scientific process trace including failures, revisions, and human interventions.** The field has produced excellent benchmarks for isolated scientific capabilities (ScienceAgentBench for data-driven tasks, SciCode for scientific coding, DiscoveryBench for hypothesis generation) and impressive autonomous systems (Coscientist, A-Lab, Robin, The AI Scientist), but evaluation remains product-focused rather than process-focused. Every benchmark measures *what* was achieved (final code, paper, or hypothesis) but never *how* the agent arrived there---which hypotheses it abandoned, which experiments failed, how it recovered from errors, and when humans had to intervene.

Meanwhile, the agent trajectory dataset community (WebArena, AgentBench, ToolBench, OpenHands) has developed rich trace formats based on the ReAct paradigm (Thought-Action-Observation triples), but exclusively for web, software engineering, and tool-use domains. None targets scientific discovery. The scientific workflow community (Coscientist, Robin, A-Lab) has built systems that execute the full scientific loop, but none publishes structured, reusable trace data. And the governance community has proposed autonomy taxonomies (L0-L5 levels) and attribution frameworks (CRediT), but notes that auditing AI decisions requires full traces that simply do not exist.

**OpenDiscoveryTrace fills the intersection of all four gaps**: it is the first dataset providing structured, machine-readable traces of AI scientific agents executing the full discovery loop across multiple domains, with explicit logging of failures, revisions, and human intervention points.

## Key Findings by Facet

### Facet 1: AI Science Benchmarks
- **ScienceAgentBench** (Chen et al., 2025): 102 tasks across 4 disciplines; evaluates final Python programs only.
- **The AI Scientist** (Lu et al., 2024): Full paper generation pipeline; no structured process traces published.
- **DiscoveryBench** (Majumder et al., 2024): 264 real + 903 synthetic discovery tasks; evaluates final hypothesis quality only.
- **PaperBench** (Starace et al., 2025): 8,316 sub-tasks from 20 ICML papers; hierarchical rubric but no temporal process.
- **ResearchBench** (Liu et al., 2025): 12 disciplines; independent sub-task outputs, no integrated iterative process.
- **Gap**: No benchmark records failure trajectories, revision dynamics, or human-AI interaction patterns.

### Facet 2: Agent Trajectory Datasets
- **ReAct** (Yao et al., 2023): Established Thought-Action-Observation format; de facto standard.
- **ToolBench** (Qin et al., 2024): Most explicitly structured trace format with DFS exploration paths.
- **MINT** (Wang et al., 2024): Multi-turn with simulated human feedback; 2-17% improvement from feedback.
- **OpenHands** (Wang et al., 2025): 67K+ trajectories on HuggingFace; typed event streams.
- **Gap**: All focused on web/software; zero scientific discovery trajectory datasets exist.

### Facet 3: Scientific Workflow Systems
- **Coscientist** (Boiko et al., 2023, Nature): Full closed-loop chemical synthesis with GPT-4 + robotics.
- **A-Lab** (Szymanski et al., 2023, Nature): Autonomous materials synthesis; 41/58 targets succeeded.
- **Robin** (Ghareeb et al., 2025, FutureHouse): First system to autonomously discover and validate a therapeutic.
- **AI Scientist v2** (Yamada et al., 2025): First AI-generated paper accepted at peer-reviewed workshop.
- **Gap**: None publishes reusable structured trace data. Failure data is systematically omitted.

### Facet 4: Dataset Design Best Practices
- **Datasheets for Datasets** (Gebru et al., 2021): Standard documentation framework---now mandatory at NeurIPS.
- **Annotation artifacts** (Gururangan et al., 2018): Crowd-sourced annotations can introduce spurious shortcuts.
- **Inter-annotator agreement**: Cohen's kappa, Fleiss' kappa, Krippendorff's alpha are standard metrics.
- **HELM** (Liang et al., 2023): Holistic evaluation across 42 scenarios, 7 metric dimensions.
- **Gap**: No annotation standards exist for multi-step agent trajectory quality assessment.

### Facet 5: Governance and Autonomy
- **L0-L5 autonomy levels** (Tom et al., 2024; Hung et al., 2024): Analogous to self-driving car levels.
- **CRediT taxonomy** (Brand et al., 2015): 14 contributor roles; needs AI-specific extension.
- **Robot Scientist Adam** (King et al., 2009, Science): First autonomous hypothesis generation system.
- **Gap**: Auditing AI decisions requires full traces that do not exist. No empirical validation of autonomy frameworks.

## Identified Gaps & Opportunities
1. No published dataset of structured AI scientific reasoning traces (including failures and revisions)
2. No standardized schema for representing scientific discovery process trajectories
3. No benchmark tasks that evaluate process quality rather than just final outputs
4. No systematic logging of human intervention points in AI-science systems
5. Governance and auditing frameworks lack the trace data needed to be operationalized
6. Autonomy level taxonomies have not been empirically validated against real system traces

## Complete References
[See individual facet outputs for full BibTeX; consolidated in paper]
