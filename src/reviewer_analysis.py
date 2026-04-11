"""
OpenDiscoveryTrace: Extended Analysis for Reviewer Response
Computes: benchmark baselines, failure taxonomy, token stats, matched comparisons, mixed-effects
"""
import json, glob, os, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'font.family':'serif','figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})

TRAJ_DIR = "trajectories"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {'gpt-5.4':'#0072B2','claude-opus-4.6':'#D55E00','gemini-3.1-pro':'#009E73'}
MLABELS = {'gpt-5.4':'GPT-5.4','claude-opus-4.6':'Claude Opus 4.6','gemini-3.1-pro':'Gemini 3.1 Pro'}

# ── Load ────────────────────────────────────────────────────────────────────
def load_all():
    trajs = []
    for f in sorted(glob.glob(f"{TRAJ_DIR}/*.json")):
        trajs.append(json.load(open(f)))
    return trajs

def to_df(trajs):
    rows = []
    for t in trajs:
        m = t.get("metadata",{})
        o = t.get("outcome",{})
        steps = t.get("trajectory",[])
        # Failure taxonomy
        error_types = []
        for s in steps:
            raw = s.get("raw_response","").lower()
            e = s.get("error",{})
            if e.get("occurred"):
                msg = e.get("message","").lower()
                if "hallucin" in raw: error_types.append("hallucination")
                elif "tool" in e.get("type","") or "api" in e.get("type",""): error_types.append("tool_misuse")
                elif "timeout" in msg: error_types.append("timeout")
                elif "not found" in msg or "404" in msg: error_types.append("resource_not_found")
                elif "rate" in msg or "429" in msg: error_types.append("rate_limit")
                else: error_types.append("other_error")
            if s.get("revision_trigger"):
                if "error" in str(s.get("revision_trigger","")).lower(): error_types.append("reasoning_error")
        # Token estimation
        total_chars = sum(len(s.get("raw_response","")) for s in steps)
        total_obs_chars = sum(len(str(s.get("observation",""))) for s in steps)
        # Tools used
        tools_used = [s.get("action",{}).get("tool","") for s in steps if s.get("action",{}).get("tool")]
        tool_types = Counter(tools_used)
        # Phases
        phases = [s.get("phase","unknown") for s in steps]
        rows.append({
            "trajectory_id": t.get("trajectory_id",""),
            "task_id": t.get("task_id",""),
            "domain": t.get("domain",""),
            "difficulty": t.get("difficulty",""),
            "model": t.get("model",""),
            "success": o.get("success"),
            "has_claim": o.get("final_claim") is not None,
            "confidence": o.get("confidence"),
            "total_steps": m.get("total_steps",0),
            "total_tool_calls": m.get("total_tool_calls",0),
            "total_failures": m.get("total_failures",0),
            "total_revisions": m.get("total_revisions",0),
            "wall_time": m.get("wall_time_seconds",0),
            "max_steps_reached": m.get("max_steps_reached",False),
            "tokens_est": m.get("total_tokens_est",0),
            "total_chars_response": total_chars,
            "total_chars_observation": total_obs_chars,
            "unique_tools": len(set(tools_used)),
            "tool_types": dict(tool_types),
            "error_types": error_types,
            "n_error_types": len(set(error_types)),
            "unique_phases": len(set(phases)),
            "recovery_attempted": o.get("recovery_attempted", False),
        })
    return pd.DataFrame(rows)

# ── 1. BENCHMARK TASK BASELINES ─────────────────────────────────────────────

def benchmark_task1_outcome_prediction(df):
    """Task 1: Trajectory Outcome Prediction - predict success from early-step features."""
    print("\n=== BENCHMARK TASK 1: Trajectory Outcome Prediction ===")
    eval_df = df[df["success"].notna()].copy()
    if len(eval_df) < 20:
        print(f"  Only {len(eval_df)} evaluable trajectories, skipping")
        return {}
    
    features = ["total_steps","total_tool_calls","total_failures","total_revisions",
                 "unique_tools","unique_phases","total_chars_response","n_error_types"]
    X = eval_df[features].fillna(0).values
    y = eval_df["success"].astype(int).values
    groups = eval_df["task_id"].values
    
    results = {}
    # Majority baseline
    majority = max(y.mean(), 1-y.mean())
    results["majority_baseline"] = round(majority, 3)
    print(f"  Majority baseline: {majority:.3f}")
    
    for name, clf in [("LogisticRegression", LogisticRegression(max_iter=1000)),
                      ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
                      ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42))]:
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            results[name] = {"accuracy": round(scores.mean(),3), "acc_std": round(scores.std(),3),
                           "auroc": round(auc_scores.mean(),3), "auroc_std": round(auc_scores.std(),3)}
            print(f"  {name}: Acc={scores.mean():.3f}±{scores.std():.3f}, AUROC={auc_scores.mean():.3f}±{auc_scores.std():.3f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results[name] = {"error": str(e)}
    
    # Feature importance from RF
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        imp = dict(zip(features, [round(x,3) for x in rf.feature_importances_]))
        results["feature_importance"] = imp
        print(f"  Top features: {sorted(imp.items(), key=lambda x:-x[1])[:3]}")
    except: pass
    
    return results

def benchmark_task2_error_localization(trajs, df):
    """Task 2: Error Localization - identify which step has first error."""
    print("\n=== BENCHMARK TASK 2: Error Localization ===")
    failed_trajs = [t for t in trajs if t["metadata"]["total_failures"] > 0]
    print(f"  Failed trajectories: {len(failed_trajs)}")
    
    # Ground truth: first step with error
    first_errors = []
    for t in failed_trajs:
        for i, s in enumerate(t["trajectory"]):
            if s.get("error",{}).get("occurred"):
                first_errors.append({"traj": t["trajectory_id"], "step": i,
                                    "total_steps": len(t["trajectory"]),
                                    "relative_pos": i/max(len(t["trajectory"]),1)})
                break
    
    if not first_errors:
        print("  No errors found")
        return {}
    
    positions = [e["relative_pos"] for e in first_errors]
    results = {
        "n_failed_trajectories": len(failed_trajs),
        "mean_first_error_position": round(np.mean(positions),3),
        "median_first_error_position": round(np.median(positions),3),
        "std_first_error_position": round(np.std(positions),3),
    }
    
    # Heuristic baselines
    # Baseline 1: Always predict step 0
    acc_step0 = sum(1 for e in first_errors if e["step"]==0) / len(first_errors)
    results["baseline_always_step0"] = round(acc_step0, 3)
    
    # Baseline 2: Always predict last step
    acc_last = sum(1 for e in first_errors if e["step"]==e["total_steps"]-1) / len(first_errors)
    results["baseline_always_last"] = round(acc_last, 3)
    
    # Baseline 3: Random
    results["baseline_random"] = round(1/max(np.mean([e["total_steps"] for e in first_errors]),1), 3)
    
    print(f"  Mean first error at position: {np.mean(positions):.3f} (0=first step, 1=last step)")
    print(f"  Baseline 'always step 0': {acc_step0:.3f}")
    print(f"  Baseline 'always last': {acc_last:.3f}")
    
    return results

def benchmark_task3_claim_verification(df):
    """Task 3: Claim Verification - success rate on ground-truth tasks."""
    print("\n=== BENCHMARK TASK 3: Claim Verification ===")
    eval_df = df[df["success"].notna()].copy()
    results = {}
    for model in sorted(eval_df["model"].unique()):
        m_df = eval_df[eval_df["model"]==model]
        sr = m_df["success"].mean()
        n = len(m_df)
        # Wilson score CI
        z = 1.96
        p_hat = sr
        denom = 1 + z**2/n
        center = (p_hat + z**2/(2*n))/denom
        spread = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
        ci_low, ci_high = max(0, center-spread), min(1, center+spread)
        results[model] = {"success_rate": round(sr,3), "n": n,
                         "ci_95_low": round(ci_low,3), "ci_95_high": round(ci_high,3)}
        print(f"  {MLABELS.get(model,model)}: {sr:.3f} [{ci_low:.3f}, {ci_high:.3f}] (n={n})")
    
    # Per domain
    for domain in sorted(eval_df["domain"].unique()):
        d_df = eval_df[eval_df["domain"]==domain]
        if len(d_df) > 5:
            results[f"domain_{domain}"] = {"success_rate": round(d_df["success"].mean(),3), "n": len(d_df)}
    
    return results

def benchmark_task5_process_quality(df):
    """Task 5: Process Quality Scoring - composite metric."""
    print("\n=== BENCHMARK TASK 5: Process Quality Scoring ===")
    # Define composite quality score: weighted combination of normalized metrics
    df_copy = df.copy()
    
    # Normalize each metric 0-1 (higher is better)
    df_copy["efficiency"] = 1 - (df_copy["total_steps"] / df_copy["total_steps"].max()).clip(0,1)
    df_copy["tool_use"] = (df_copy["total_tool_calls"] / df_copy["total_tool_calls"].max()).clip(0,1)
    df_copy["low_errors"] = 1 - (df_copy["total_failures"] / max(df_copy["total_failures"].max(),1)).clip(0,1)
    df_copy["conclusion"] = df_copy["has_claim"].astype(float)
    df_copy["quality_score"] = (df_copy["efficiency"] + df_copy["tool_use"] + df_copy["low_errors"] + df_copy["conclusion"]) / 4
    
    results = {}
    for model in sorted(df_copy["model"].unique()):
        m_df = df_copy[df_copy["model"]==model]
        qs = m_df["quality_score"]
        results[model] = {"mean": round(qs.mean(),3), "std": round(qs.std(),3), "median": round(qs.median(),3)}
        print(f"  {MLABELS.get(model,model)}: {qs.mean():.3f}±{qs.std():.3f}")
    
    # Kruskal-Wallis
    groups = [df_copy[df_copy["model"]==m]["quality_score"].values for m in sorted(df_copy["model"].unique())]
    h, p = stats.kruskal(*groups)
    results["kruskal_wallis"] = {"H": round(h,3), "p": round(p,6)}
    print(f"  Kruskal-Wallis: H={h:.3f}, p={p:.6f}")
    
    return results

# ── 2. FAILURE TAXONOMY ─────────────────────────────────────────────────────

def analyze_failure_taxonomy(df):
    """Analyze 8-category failure types across models."""
    print("\n=== FAILURE TAXONOMY ANALYSIS ===")
    all_errors = defaultdict(lambda: defaultdict(int))
    total_per_model = defaultdict(int)
    
    for _, row in df.iterrows():
        model = row["model"]
        for et in row["error_types"]:
            all_errors[model][et] += 1
            total_per_model[model] += 1
    
    results = {}
    all_types = sorted(set(et for model_errors in all_errors.values() for et in model_errors))
    
    print(f"  Error types found: {all_types}")
    for model in sorted(all_errors.keys()):
        print(f"\n  {MLABELS.get(model,model)} (total errors: {total_per_model[model]}):")
        results[model] = {}
        for et in all_types:
            count = all_errors[model].get(et, 0)
            pct = count / max(total_per_model[model], 1) * 100
            results[model][et] = {"count": count, "pct": round(pct,1)}
            if count > 0:
                print(f"    {et}: {count} ({pct:.1f}%)")
    
    return results

# ── 3. TOKEN AND TOOL STATISTICS ────────────────────────────────────────────

def compute_token_stats(df, trajs):
    """Token-level and per-tool statistics."""
    print("\n=== TOKEN & TOOL STATISTICS ===")
    
    results = {
        "total_response_chars": int(df["total_chars_response"].sum()),
        "total_observation_chars": int(df["total_chars_observation"].sum()),
        "est_total_tokens": int(df["total_chars_response"].sum() / 4),  # rough char/token ratio
        "mean_response_chars_per_traj": round(df["total_chars_response"].mean(), 0),
        "mean_observation_chars_per_traj": round(df["total_chars_observation"].mean(), 0),
    }
    
    # Per-model token stats
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"]==model]
        results[f"{model}_mean_chars"] = round(m_df["total_chars_response"].mean(), 0)
        results[f"{model}_est_tokens"] = round(m_df["total_chars_response"].mean() / 4, 0)
    
    # Tool usage breakdown
    tool_counts = defaultdict(lambda: defaultdict(int))
    tool_latency = defaultdict(list)
    for t in trajs:
        model = t["model"]
        for s in t.get("trajectory", []):
            tool = s.get("action",{}).get("tool","")
            if tool:
                tool_counts[model][tool] += 1
                wt = s.get("wall_time", 0)
                if wt > 0:
                    tool_latency[tool].append(wt)
    
    results["tool_usage"] = {}
    for model in sorted(tool_counts.keys()):
        results["tool_usage"][model] = dict(tool_counts[model])
        print(f"  {MLABELS.get(model,model)} tools: {dict(tool_counts[model])}")
    
    if tool_latency:
        results["tool_latency"] = {t: {"mean": round(np.mean(v),2), "n": len(v)} 
                                   for t, v in tool_latency.items() if v}
        print(f"  Tool latencies: { {t: f'{np.mean(v):.1f}s (n={len(v)})' for t,v in tool_latency.items() if v} }")
    
    print(f"  Total est tokens: ~{results['est_total_tokens']:,}")
    print(f"  Per model chars: { {MLABELS.get(m,m): results.get(f'{m}_mean_chars',0) for m in df['model'].unique()} }")
    
    return results

# ── 4. MATCHED-TASK COMPARISONS ─────────────────────────────────────────────

def matched_task_comparisons(df):
    """Within-domain, matched-task comparisons controlling for task difficulty."""
    print("\n=== MATCHED-TASK WITHIN-DOMAIN COMPARISONS ===")
    results = {}
    
    models = sorted(df["model"].unique())
    
    for domain in sorted(df["domain"].unique()):
        d_df = df[df["domain"]==domain]
        print(f"\n  Domain: {domain}")
        
        # Find tasks present in all models
        task_model_counts = d_df.groupby("task_id")["model"].nunique()
        shared_tasks = task_model_counts[task_model_counts == len(models)].index.tolist()
        matched_df = d_df[d_df["task_id"].isin(shared_tasks)]
        
        if len(shared_tasks) < 5:
            print(f"    Only {len(shared_tasks)} shared tasks, skipping")
            continue
        
        print(f"    Shared tasks: {len(shared_tasks)}, matched trajectories: {len(matched_df)}")
        
        domain_results = {"n_shared_tasks": len(shared_tasks), "n_matched_trajectories": len(matched_df)}
        
        for metric in ["total_steps","total_tool_calls","total_failures","total_revisions"]:
            groups = [matched_df[matched_df["model"]==m][metric].values for m in models]
            valid = [g for g in groups if len(g) > 0]
            if len(valid) >= 2:
                h, p = stats.kruskal(*valid)
                domain_results[metric] = {
                    "per_model": {m: {"mean": round(matched_df[matched_df["model"]==m][metric].mean(),2),
                                     "std": round(matched_df[matched_df["model"]==m][metric].std(),2)}
                                 for m in models},
                    "H": round(h,3), "p": round(p,6)
                }
                sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                means = {MLABELS.get(m,m): round(matched_df[matched_df["model"]==m][metric].mean(),2) for m in models}
                print(f"    {metric}: {means} (H={h:.2f}, p={p:.4f} {sig})")
        
        results[domain] = domain_results
    
    # Logistic regression with domain and model as factors (for success)
    eval_df = df[df["success"].notna()].copy()
    if len(eval_df) > 20:
        print(f"\n  Logistic regression (domain × model → success):")
        eval_df["model_code"] = pd.Categorical(eval_df["model"]).codes
        eval_df["domain_code"] = pd.Categorical(eval_df["domain"]).codes
        X = eval_df[["model_code","domain_code","total_steps","total_failures"]].values
        y = eval_df["success"].astype(int).values
        try:
            lr = LogisticRegression(max_iter=1000).fit(X, y)
            acc = lr.score(X, y)
            results["logistic_regression"] = {"accuracy": round(acc,3), "n": len(eval_df),
                                             "coefs": {n: round(c,3) for n,c in zip(["model","domain","steps","failures"], lr.coef_[0])}}
            print(f"    Accuracy: {acc:.3f}, Coefficients: {results['logistic_regression']['coefs']}")
        except Exception as e:
            print(f"    Failed: {e}")
    
    return results

# ── 5. DIFFICULTY ANALYSIS ──────────────────────────────────────────────────

def difficulty_analysis(df):
    """Per-difficulty, per-model detailed stats."""
    print("\n=== DIFFICULTY × MODEL ANALYSIS ===")
    results = {}
    for diff in ["easy","medium","hard"]:
        d_df = df[df["difficulty"]==diff]
        if len(d_df) == 0: continue
        results[diff] = {"n": len(d_df)}
        for model in sorted(d_df["model"].unique()):
            m_df = d_df[d_df["model"]==model]
            results[diff][model] = {
                "n": len(m_df),
                "mean_steps": round(m_df["total_steps"].mean(),2),
                "mean_errors": round(m_df["total_failures"].mean(),2),
                "mean_revisions": round(m_df["total_revisions"].mean(),2),
                "conclusion_rate": round(m_df["has_claim"].mean(),3),
            }
        print(f"  {diff}: n={len(d_df)}, steps={d_df['total_steps'].mean():.1f}, errors={d_df['total_failures'].mean():.1f}")
    return results

# ── 6. FIGURE: Failure taxonomy ─────────────────────────────────────────────

def fig_failure_taxonomy(df):
    """Figure: Failure type distribution across models."""
    all_errors = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        for et in row["error_types"]:
            all_errors[row["model"]][et] += 1
    
    models = sorted(all_errors.keys())
    all_types = sorted(set(et for m in all_errors.values() for et in m))
    if not all_types:
        print("  No error types to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(all_types))
    width = 0.25
    for i, model in enumerate(models):
        total = sum(all_errors[model].values())
        vals = [all_errors[model].get(t, 0) / max(total, 1) for t in all_types]
        ax.bar(x + i*width, vals, width, label=MLABELS.get(model, model),
              color=COLORS.get(model, '#999'), edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Proportion of Errors')
    ax.set_title('Error Type Distribution Across Models', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace('_',' ').title() for t in all_types], rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_failure_taxonomy.pdf")
    plt.savefig(f"{FIG_DIR}/fig6_failure_taxonomy.png")
    plt.close()
    print("  Saved fig6_failure_taxonomy")

# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("OpenDiscoveryTrace: Extended Reviewer Analysis")
    print("="*60)
    
    trajs = load_all()
    df = to_df(trajs)
    print(f"Loaded {len(trajs)} trajectories, {df['model'].nunique()} models")
    
    all_results = {}
    
    # Benchmark baselines
    all_results["task1_outcome_prediction"] = benchmark_task1_outcome_prediction(df)
    all_results["task2_error_localization"] = benchmark_task2_error_localization(trajs, df)
    all_results["task3_claim_verification"] = benchmark_task3_claim_verification(df)
    all_results["task5_process_quality"] = benchmark_task5_process_quality(df)
    
    # Failure taxonomy
    all_results["failure_taxonomy"] = analyze_failure_taxonomy(df)
    
    # Token stats
    all_results["token_stats"] = compute_token_stats(df, trajs)
    
    # Matched comparisons
    all_results["matched_comparisons"] = matched_task_comparisons(df)
    
    # Difficulty analysis
    all_results["difficulty_analysis"] = difficulty_analysis(df)
    
    # Failure taxonomy figure
    fig_failure_taxonomy(df)
    
    # Cliff's delta for all pairwise comparisons
    print("\n=== CLIFF'S DELTA EFFECT SIZES ===")
    all_results["cliffs_delta"] = {}
    models = sorted(df["model"].unique())
    for metric in ["total_steps","total_tool_calls","total_failures","total_revisions"]:
        all_results["cliffs_delta"][metric] = {}
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                x = df[df["model"]==m1][metric].values
                y = df[df["model"]==m2][metric].values
                # Cliff's delta
                n = len(x) * len(y)
                if n > 0:
                    more = sum(1 for xi in x for yi in y if xi > yi)
                    less = sum(1 for xi in x for yi in y if xi < yi)
                    delta = (more - less) / n
                else:
                    delta = 0
                pair = f"{m1}_vs_{m2}"
                all_results["cliffs_delta"][metric][pair] = round(delta, 3)
                size = "large" if abs(delta)>0.474 else "medium" if abs(delta)>0.33 else "small" if abs(delta)>0.147 else "negligible"
                print(f"  {metric} {MLABELS.get(m1,m1)} vs {MLABELS.get(m2,m2)}: δ={delta:.3f} ({size})")
    
    # Save
    with open("reviewer_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to reviewer_analysis_results.json")
    print("Done!")

if __name__ == "__main__":
    main()
