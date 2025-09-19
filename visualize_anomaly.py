#!/usr/bin/env python3
"""
Visualize the depth 2.0 anomaly with scientific rigor.

Creates publication-quality figures showing the unexpected peak.
Based on data visualization best practices (Tufte, 1983; Cleveland, 1985).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(filepath='data/processed/results_20250919_103108.json'):
    """Load experimental results."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comprehensive_plot(data):
    """
    Create multi-panel figure showing the anomaly.
    
    Following principles from Cleveland & McGill (1984, JASA) on
    graphical perception and statistical graphics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    models = ['gpt-3.5', 'claude-3-haiku']
    colors = {'gpt-3.5': '#1f77b4', 'claude-3-haiku': '#ff7f0e'}
    
    # Panel 1: Raw scatter plot with means
    ax1 = axes[0, 0]
    for model in models:
        depths = []
        scores = []
        for trial in data['by_model'][model]:
            depths.append(trial['depth'])
            scores.append(trial['scores']['total'])
        
        # Plot individual points
        ax1.scatter(depths, scores, alpha=0.3, s=20, 
                   color=colors[model], label=f'{model} (raw)')
        
        # Calculate and plot means
        unique_depths = sorted(set(depths))
        means = []
        for d in unique_depths:
            d_scores = [s for dep, s in zip(depths, scores) if dep == d]
            means.append(np.mean(d_scores))
        
        ax1.plot(unique_depths, means, 'o-', color=colors[model], 
                linewidth=2, markersize=8, label=f'{model} (mean)')
    
    ax1.set_xlabel('Temporal Depth')
    ax1.set_ylabel('Coherence Score')
    ax1.set_title('A. Raw Data Showing Depth 2.0 Peak')
    ax1.legend(loc='upper right')
    ax1.axvline(x=2.0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Violin plot at key depths
    ax2 = axes[0, 1]
    key_depths = [1.0, 2.0, 3.0]
    
    for i, depth in enumerate(key_depths):
        for j, model in enumerate(models):
            scores = [t['scores']['total'] for t in data['by_model'][model] 
                     if t['depth'] == depth]
            
            positions = [i + j*0.35 - 0.175]
            parts = ax2.violinplot([scores], positions=positions, 
                                  widths=0.3, showmeans=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor(colors[model])
                pc.set_alpha(0.7)
    
    ax2.set_xticks(range(len(key_depths)))
    ax2.set_xticklabels([f'Depth {d}' for d in key_depths])
    ax2.set_ylabel('Coherence Score')
    ax2.set_title('B. Distribution at Key Depths')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Difference from expected exponential
    ax3 = axes[0, 2]
    for model in models:
        depths = []
        scores = []
        for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            d_scores = [t['scores']['total'] for t in data['by_model'][model] 
                       if t['depth'] == d]
            if d_scores:
                depths.append(d)
                scores.append(np.mean(d_scores))
        
        # Fit exponential excluding depth 2.0
        depths_no2 = [d for d in depths if d != 2.0]
        scores_no2 = [s for d, s in zip(depths, scores) if d != 2.0]
        
        # Simple linear fit in log space
        log_scores = np.log(np.array(scores_no2) + 0.01)  # Add small constant to avoid log(0)
        z = np.polyfit(depths_no2, log_scores, 1)
        
        # Calculate expected values
        expected = np.exp(np.poly1d(z)(depths))
        residuals = np.array(scores) - expected
        
        ax3.bar([d + (0.2 if model == 'gpt-3.5' else -0.2) for d in depths], 
               residuals, width=0.35, label=model, color=colors[model], alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Temporal Depth')
    ax3.set_ylabel('Residual (Actual - Expected)')
    ax3.set_title('C. Deviation from Exponential Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Subscore breakdown at depth 2.0
    ax4 = axes[1, 0]
    subscores = ['temporal_ordering', 'causal_maintenance', 'depth_accuracy', 'transition_smoothness']
    
    for i, model in enumerate(models):
        depth2_trials = [t for t in data['by_model'][model] if t['depth'] == 2.0]
        
        subscore_means = []
        for sub in subscores:
            values = [t['scores'][sub] for t in depth2_trials]
            subscore_means.append(np.mean(values))
        
        x = np.arange(len(subscores))
        ax4.bar(x + i*0.35 - 0.175, subscore_means, 0.35, 
               label=model, color=colors[model], alpha=0.7)
    
    ax4.set_xticks(range(len(subscores)))
    ax4.set_xticklabels(['Temporal\nOrdering', 'Causal\nMaintenance', 
                         'Depth\nAccuracy', 'Transition\nSmoothness'], rotation=45)
    ax4.set_ylabel('Mean Subscore')
    ax4.set_title('D. Subscore Analysis at Depth 2.0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Statistical significance heatmap
    ax5 = axes[1, 1]
    depths_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for m_idx, model in enumerate(models):
        p_values = []
        for i, d1 in enumerate(depths_to_test):
            row = []
            for j, d2 in enumerate(depths_to_test):
                if i >= j:
                    row.append(np.nan)
                else:
                    scores1 = [t['scores']['total'] for t in data['by_model'][model] 
                              if t['depth'] == d1]
                    scores2 = [t['scores']['total'] for t in data['by_model'][model] 
                              if t['depth'] == d2]
                    if scores1 and scores2:
                        _, p = stats.ttest_ind(scores1, scores2)
                        row.append(p)
                    else:
                        row.append(np.nan)
            p_values.append(row)
        
        if m_idx == 0:
            im = ax5.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
            ax5.set_title(f'E. P-values: {model}')
        else:
            ax6 = axes[1, 2]
            im = ax6.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
            ax6.set_title(f'F. P-values: {model}')
            ax6.set_xticks(range(len(depths_to_test)))
            ax6.set_xticklabels(depths_to_test)
            ax6.set_yticks(range(len(depths_to_test)))
            ax6.set_yticklabels(depths_to_test)
            plt.colorbar(im, ax=ax6)
    
    ax5.set_xticks(range(len(depths_to_test)))
    ax5.set_xticklabels(depths_to_test)
    ax5.set_yticks(range(len(depths_to_test)))
    ax5.set_yticklabels(depths_to_test)
    
    plt.suptitle('Temporal Coherence Anomaly: Unexpected Peak at Depth 2.0', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = 'results/figures/depth2_anomaly.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.show()

def calculate_peak_statistics(data):
    """
    Calculate statistical properties of the depth 2.0 peak.
    
    Following recommendations from Cumming (2014, Psychological Science)
    on reporting effect sizes and confidence intervals.
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF DEPTH 2.0 PEAK")
    print("="*60)
    
    models = ['gpt-3.5', 'claude-3-haiku']
    
    for model in models:
        print(f"\n{model.upper()}:")
        
        # Get scores for key depths
        scores_1 = [t['scores']['total'] for t in data['by_model'][model] if t['depth'] == 1.0]
        scores_2 = [t['scores']['total'] for t in data['by_model'][model] if t['depth'] == 2.0]
        scores_3 = [t['scores']['total'] for t in data['by_model'][model] if t['depth'] == 3.0]
        
        # Calculate effect sizes (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
        
        d_12 = cohens_d(scores_2, scores_1)
        d_23 = cohens_d(scores_2, scores_3)
        
        print(f"  Mean scores: 1.0={np.mean(scores_1):.3f}, 2.0={np.mean(scores_2):.3f}, 3.0={np.mean(scores_3):.3f}")
        print(f"  Effect size (d) depth 2 vs 1: {d_12:.3f}")
        print(f"  Effect size (d) depth 2 vs 3: {d_23:.3f}")
        
        # Bootstrap confidence interval for depth 2.0 mean
        from scipy.stats import bootstrap
        rng = np.random.default_rng(42)
        res = bootstrap((scores_2,), np.mean, n_resamples=10000, random_state=rng)
        print(f"  95% CI for depth 2.0 mean: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]")
        
        # Test for non-monotonicity
        all_depths = sorted(set(t['depth'] for t in data['by_model'][model]))
        mean_trajectory = []
        for d in all_depths:
            d_scores = [t['scores']['total'] for t in data['by_model'][model] if t['depth'] == d]
            mean_trajectory.append(np.mean(d_scores) if d_scores else 0)
        
        # Count violations of monotonic decrease
        violations = 0
        for i in range(1, len(mean_trajectory)):
            if mean_trajectory[i] > mean_trajectory[i-1]:
                violations += 1
        
        print(f"  Monotonicity violations: {violations}/{len(mean_trajectory)-1}")

if __name__ == "__main__":
    data = load_data()
    create_comprehensive_plot(data)
    calculate_peak_statistics(data)