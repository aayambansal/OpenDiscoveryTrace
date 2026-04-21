"""
OpenDiscoveryTrace: Analysis Pipeline
Computes dataset statistics, inter-model comparisons, and generates figures.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Configuration ───────────────────────────────────────────────────────────
TRAJ_DIR = "trajectories"
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'gpt-5.4': '#0072B2',
    'claude-opus-4.6': '#D55E00',
    'gemini-3.1-pro': '#009E73',
    # Legacy
    'gpt-4o': '#0072B2',
    'claude-sonnet-4': '#D55E00',
    'gemini-2.5-pro': '#009E73',
}
MODEL_LABELS = {
    'gpt-5.4': 'GPT-5.4',
    'claude-opus-4.6': 'Claude Opus 4.6',
    'gemini-3.1-pro': 'Gemini 3.1 Pro',
    # Legacy
    'gpt-4o': 'GPT-4o',
    'claude-sonnet-4': 'Claude Sonnet 4',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
}
DOMAIN_LABELS = {
    'drug_discovery': 'Drug Discovery',
    'materials_science': 'Materials Science',
    'genomics': 'Genomics',
    'literature': 'Literature'
}

# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_trajectories():
    """Load all trajectory JSON files into a list."""
    trajectories = []
    for fpath in sorted(glob.glob(f"{TRAJ_DIR}/*.json")):
        with open(fpath) as f:
            trajectories.append(json.load(f))
    return trajectories


def trajectories_to_dataframe(trajectories):
    """Convert trajectories to a pandas DataFrame for analysis."""
    rows = []
    for t in trajectories:
        meta = t.get("metadata", {})
        outcome = t.get("outcome", {})
        
        # Count phases
        phases = [s.get("phase", "unknown") for s in t.get("trajectory", [])]
        unique_phases = len(set(phases))
        
        # Count error types
        errors = [s.get("error", {}) for s in t.get("trajectory", []) if s.get("error", {}).get("occurred")]
        error_types = [e.get("type", "unknown") for e in errors]
        
        # Detect revision steps
        revisions = sum(1 for s in t.get("trajectory", []) if s.get("revision_trigger"))
        
        # Calculate tool diversity
        tools_used = [s.get("action", {}).get("tool", "") for s in t.get("trajectory", []) if s.get("action", {}).get("tool")]
        unique_tools = len(set(tools_used))
        
        rows.append({
            "trajectory_id": t.get("trajectory_id", ""),
            "task_id": t.get("task_id", ""),
            "domain": t.get("domain", ""),
            "difficulty": t.get("difficulty", ""),
            "model": t.get("model", ""),
            "success": outcome.get("success"),
            "has_claim": outcome.get("final_claim") is not None,
            "confidence": outcome.get("confidence"),
            "total_steps": meta.get("total_steps", 0),
            "total_tool_calls": meta.get("total_tool_calls", 0),
            "total_failures": meta.get("total_failures", 0),
            "total_revisions": meta.get("total_revisions", 0),
            "wall_time": meta.get("wall_time_seconds", 0),
            "max_steps_reached": meta.get("max_steps_reached", False),
            "unique_phases": unique_phases,
            "unique_tools": unique_tools,
            "recovery_attempted": outcome.get("recovery_attempted", False),
            "error_types": ",".join(error_types) if error_types else "",
            "tokens_est": meta.get("total_tokens_est", 0),
        })
    
    return pd.DataFrame(rows)


# ── Analysis Functions ──────────────────────────────────────────────────────

def compute_summary_stats(df):
    """Compute overall dataset statistics."""
    stats_dict = {
        "total_trajectories": len(df),
        "models": df["model"].nunique(),
        "domains": df["domain"].nunique(),
        "tasks": df["task_id"].nunique(),
        "success_rate": df["success"].mean() if df["success"].notna().any() else None,
        "has_claim_rate": df["has_claim"].mean(),
        "mean_steps": df["total_steps"].mean(),
        "median_steps": df["total_steps"].median(),
        "mean_tool_calls": df["total_tool_calls"].mean(),
        "mean_failures": df["total_failures"].mean(),
        "mean_revisions": df["total_revisions"].mean(),
        "failure_rate": (df["total_failures"] > 0).mean(),
        "revision_rate": (df["total_revisions"] > 0).mean(),
        "mean_wall_time": df["wall_time"].mean(),
        "max_steps_rate": df["max_steps_reached"].mean(),
    }
    return stats_dict


def compute_model_comparison(df):
    """Compare metrics across models."""
    metrics = ["success", "total_steps", "total_tool_calls", "total_failures", "total_revisions", "wall_time", "has_claim"]
    results = {}
    
    for metric in metrics:
        model_groups = {}
        for model in df["model"].unique():
            model_data = df[df["model"] == model][metric].dropna()
            model_groups[model] = model_data
            
        results[metric] = {
            "per_model": {m: {"mean": float(v.mean()), "std": float(v.std()), "median": float(v.median()), "n": len(v)} 
                         for m, v in model_groups.items()},
        }
        
        # Kruskal-Wallis if 3+ models with data
        valid_groups = [v.values for v in model_groups.values() if len(v) > 0]
        if len(valid_groups) >= 2:
            try:
                h_stat, p_val = stats.kruskal(*valid_groups)
                results[metric]["kruskal_wallis"] = {"H": float(h_stat), "p": float(p_val)}
            except:
                results[metric]["kruskal_wallis"] = {"H": None, "p": None}
    
    return results


# ── Figure Generation ───────────────────────────────────────────────────────

def fig1_dataset_overview(df):
    """Figure 1: Dataset overview - trajectories by domain, difficulty, model."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # (a) By domain
    domain_counts = df.groupby(["domain", "model"]).size().unstack(fill_value=0)
    domain_counts = domain_counts.rename(index=DOMAIN_LABELS, columns=MODEL_LABELS)
    domain_counts.plot(kind="bar", ax=axes[0], color=[COLORS[m] for m in df["model"].unique()], edgecolor='black', linewidth=0.5)
    axes[0].set_title("(a) Trajectories by Domain", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend(fontsize=7)
    
    # (b) By difficulty
    diff_counts = df.groupby(["difficulty", "model"]).size().unstack(fill_value=0)
    diff_order = ["easy", "medium", "hard"]
    diff_counts = diff_counts.reindex([d for d in diff_order if d in diff_counts.index])
    diff_counts = diff_counts.rename(columns=MODEL_LABELS)
    diff_counts.plot(kind="bar", ax=axes[1], color=[COLORS[m] for m in df["model"].unique()], edgecolor='black', linewidth=0.5)
    axes[1].set_title("(b) Trajectories by Difficulty", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(fontsize=7)
    
    # (c) Success rate by model
    success_rates = df.groupby("model")["has_claim"].mean()
    models = list(success_rates.index)
    bars = axes[2].bar(
        [MODEL_LABELS.get(m, m) for m in models],
        [success_rates[m] for m in models],
        color=[COLORS.get(m, '#999999') for m in models],
        edgecolor='black', linewidth=0.5
    )
    axes[2].set_title("(c) Conclusion Rate by Model", fontweight='bold')
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1.1)
    for bar, m in zip(bars, models):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{success_rates[m]:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig1_dataset_overview.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig1_dataset_overview.png")
    plt.close()
    print("  Saved fig1_dataset_overview")


def fig2_process_metrics(df):
    """Figure 2: Process-level metrics comparison across models."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    models = sorted(df["model"].unique())
    metrics = [
        ("total_steps", "Steps per Trajectory", "(a)"),
        ("total_tool_calls", "Tool Calls per Trajectory", "(b)"),
        ("total_failures", "Errors per Trajectory", "(c)"),
        ("total_revisions", "Revisions per Trajectory", "(d)")
    ]
    
    for ax, (metric, label, panel) in zip(axes, metrics):
        data = [df[df["model"] == m][metric].values for m in models]
        bp = ax.boxplot(data, labels=[MODEL_LABELS.get(m, m) for m in models],
                       patch_artist=True, widths=0.6, showfliers=True,
                       flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, m in zip(bp['boxes'], models):
            patch.set_facecolor(COLORS.get(m, '#999999'))
            patch.set_alpha(0.7)
        ax.set_title(f"{panel} {label}", fontweight='bold', fontsize=10)
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig2_process_metrics.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig2_process_metrics.png")
    plt.close()
    print("  Saved fig2_process_metrics")


def fig3_phase_distribution(df, trajectories):
    """Figure 3: Distribution of scientific phases across models."""
    phase_map = {
        'literature': 'Literature Review',
        'hypothesis': 'Hypothesis',
        'experiment': 'Experiment Design',
        'execution': 'Execution',
        'analysis': 'Analysis',
        'conclusion': 'Conclusion',
    }
    
    model_phases = defaultdict(lambda: defaultdict(int))
    for t in trajectories:
        model = t.get("model", "")
        for step in t.get("trajectory", []):
            phase_raw = step.get("phase", "unknown").lower()
            matched = False
            for key, label in phase_map.items():
                if key in phase_raw:
                    model_phases[model][label] += 1
                    matched = True
                    break
            if not matched:
                model_phases[model]["Other"] += 1
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    all_phases = ['Literature Review', 'Hypothesis', 'Experiment Design', 'Execution', 'Analysis', 'Conclusion', 'Other']
    models = sorted(model_phases.keys())
    
    x = np.arange(len(all_phases))
    width = 0.25
    
    for i, model in enumerate(models):
        total = sum(model_phases[model].values())
        values = [model_phases[model].get(p, 0) / max(total, 1) for p in all_phases]
        ax.bar(x + i * width, values, width, label=MODEL_LABELS.get(model, model),
              color=COLORS.get(model, '#999999'), edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('Scientific Phase')
    ax.set_ylabel('Proportion of Steps')
    ax.set_title('Distribution of Scientific Phases Across Models', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_phases, rotation=30, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig3_phase_distribution.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig3_phase_distribution.png")
    plt.close()
    print("  Saved fig3_phase_distribution")


def fig4_failure_analysis(df):
    """Figure 4: Failure analysis - rates by domain and difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    models = sorted(df["model"].unique())
    
    # (a) Failure rate by domain
    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        domain_failure = model_df.groupby("domain").apply(lambda x: (x["total_failures"] > 0).mean())
        domains = [DOMAIN_LABELS.get(d, d) for d in domain_failure.index]
        axes[0].plot(domains, domain_failure.values, 'o-', color=COLORS.get(model, '#999999'),
                    label=MODEL_LABELS.get(model, model), markersize=6, linewidth=1.5)
    
    axes[0].set_title("(a) Error Rate by Domain", fontweight='bold')
    axes[0].set_ylabel("Proportion with Errors")
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    
    # (b) Failure rate by difficulty
    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        diff_failure = model_df.groupby("difficulty").apply(lambda x: (x["total_failures"] > 0).mean())
        diff_order = ["easy", "medium", "hard"]
        diff_failure = diff_failure.reindex([d for d in diff_order if d in diff_failure.index])
        axes[1].plot(diff_failure.index, diff_failure.values, 's-', color=COLORS.get(model, '#999999'),
                    label=MODEL_LABELS.get(model, model), markersize=6, linewidth=1.5)
    
    axes[1].set_title("(b) Error Rate by Difficulty", fontweight='bold')
    axes[1].set_ylabel("Proportion with Errors")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig4_failure_analysis.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig4_failure_analysis.png")
    plt.close()
    print("  Saved fig4_failure_analysis")


def fig5_trace_schema(df):
    """Figure 5: Example trace visualization showing the scientific workflow."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    phases = ['Literature\nReview', 'Hypothesis', 'Experiment\nDesign', 'Execution', 'Analysis', 'Conclusion']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for i, (phase, color) in enumerate(zip(phases, colors)):
        rect = plt.Rectangle((i * 1.5, 0.2), 1.2, 0.6, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(i * 1.5 + 0.6, 0.5, phase, ha='center', va='center', fontsize=8, fontweight='bold')
        if i < len(phases) - 1:
            ax.annotate('', xy=((i+1) * 1.5, 0.5), xytext=(i * 1.5 + 1.2, 0.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add failure/revision loop
    ax.annotate('', xy=(3 * 1.5, 0.2), xytext=(4.5 * 1.5, 0.2),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='dashed',
                              connectionstyle='arc3,rad=0.5'))
    ax.text(5.0, -0.1, 'Failure → Revision', color='red', fontsize=8, ha='center', fontstyle='italic')
    
    ax.set_xlim(-0.3, 9.5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_title('OpenDiscoveryTrace: Scientific Workflow Schema', fontweight='bold', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/fig5_trace_schema.pdf")
    plt.savefig(f"{FIGURE_DIR}/fig5_trace_schema.png")
    plt.close()
    print("  Saved fig5_trace_schema")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("OpenDiscoveryTrace Analysis Pipeline")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading trajectories...")
    trajectories = load_all_trajectories()
    print(f"   Loaded {len(trajectories)} trajectories")
    
    df = trajectories_to_dataframe(trajectories)
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Domains: {df['domain'].unique().tolist()}")
    
    # Summary statistics
    print("\n2. Computing summary statistics...")
    summary = compute_summary_stats(df)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")
    
    # Model comparison
    print("\n3. Computing model comparison...")
    comparison = compute_model_comparison(df)
    for metric, data in comparison.items():
        print(f"\n   {metric}:")
        for model, stats_data in data.get("per_model", {}).items():
            print(f"     {MODEL_LABELS.get(model, model)}: mean={stats_data['mean']:.3f}, std={stats_data['std']:.3f}, n={stats_data['n']}")
        kw = data.get("kruskal_wallis", {})
        if kw.get("p") is not None:
            sig = "***" if kw["p"] < 0.001 else "**" if kw["p"] < 0.01 else "*" if kw["p"] < 0.05 else "ns"
            print(f"     Kruskal-Wallis: H={kw['H']:.3f}, p={kw['p']:.4f} {sig}")
    
    # Generate figures
    print("\n4. Generating figures...")
    fig1_dataset_overview(df)
    fig2_process_metrics(df)
    fig3_phase_distribution(df, trajectories)
    fig4_failure_analysis(df)
    fig5_trace_schema(df)
    
    # Save analysis results
    print("\n5. Saving analysis results...")
    results = {
        "summary": summary,
        "model_comparison": comparison,
        "per_model_per_domain": {},
        "per_difficulty": {}
    }
    
    for model in df["model"].unique():
        for domain in df["domain"].unique():
            subset = df[(df["model"] == model) & (df["domain"] == domain)]
            if len(subset) > 0:
                key = f"{model}_{domain}"
                results["per_model_per_domain"][key] = {
                    "n": len(subset),
                    "success_rate": float(subset["success"].mean()) if subset["success"].notna().any() else None,
                    "has_claim_rate": float(subset["has_claim"].mean()),
                    "mean_steps": float(subset["total_steps"].mean()),
                    "mean_failures": float(subset["total_failures"].mean()),
                }
    
    for diff in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == diff]
        if len(subset) > 0:
            results["per_difficulty"][diff] = {
                "n": len(subset),
                "success_rate": float(subset["success"].mean()) if subset["success"].notna().any() else None,
                "has_claim_rate": float(subset["has_claim"].mean()),
                "mean_steps": float(subset["total_steps"].mean()),
                "mean_failures": float(subset["total_failures"].mean()),
            }
    
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   Results saved to analysis_results.json")
    print(f"   Figures saved to {FIGURE_DIR}/")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
