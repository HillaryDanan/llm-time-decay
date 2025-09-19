"""
Statistical analysis for LLM time decay experiments.

Core analyses:
1. Exponential decay fitting
2. Discrete vs continuous hypothesis testing
3. Bootstrap confidence intervals
4. Effect size calculations
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import ANALYSIS, OUTPUT


@dataclass
class DecayFitResult:
    """Results from exponential decay fitting."""
    r_squared: float
    C0: float  # Initial coherence
    tau: float  # Decay constant
    residuals: np.ndarray
    confidence_interval: Tuple[float, float]
    is_exponential: bool  # R² > threshold


class TemporalAnalyzer:
    """Analyze temporal decay patterns in LLM responses."""
    
    def __init__(self, results_file: Optional[str] = None):
        """
        Initialize analyzer.
        
        Args:
            results_file: Path to processed results JSON
        """
        self.results = None
        if results_file:
            self.load_results(results_file)
    
    def load_results(self, filepath: str):
        """Load experimental results."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded results from {filepath}")
        print(f"Models: {self.results['metadata']['models']}")
        print(f"Depths: {self.results['metadata']['depths']}")
    
    def fit_exponential_decay(self, model_name: str) -> DecayFitResult:
        """
        Fit exponential decay model to data.
        
        Mathematical model:
            C(d) = C₀ × e^(-d/τ)
        
        Based on temporal decay theory (Brown, 1958, Psych Review).
        """
        if not self.results:
            raise ValueError("No results loaded")
        
        # Extract data for model
        model_data = self.results['by_model'].get(model_name, [])
        if not model_data:
            raise ValueError(f"No data for model: {model_name}")
        
        # Aggregate scores by depth
        depth_scores = {}
        for trial in model_data:
            depth = trial['depth']
            score = trial['scores']['total']
            if depth not in depth_scores:
                depth_scores[depth] = []
            depth_scores[depth].append(score)
        
        # Calculate means
        depths = np.array(sorted(depth_scores.keys()))
        means = np.array([np.mean(depth_scores[d]) for d in depths])
        
        # Define exponential decay function
        def exponential_decay(d, C0, tau):
            return C0 * np.exp(-d/tau)
        
        # Fit the model
        try:
            params, covariance = curve_fit(
                exponential_decay, 
                depths, 
                means,
                p0=[0.6, 2.0],  # Initial guesses
                bounds=([0, 0.1], [1, 10])  # Bounds for C0 and tau
            )
            C0, tau = params
            
            # Calculate R²
            predicted = exponential_decay(depths, C0, tau)
            residuals = means - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((means - np.mean(means))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Bootstrap confidence interval for R²
            ci = self._bootstrap_r_squared(depth_scores, depths)
            
            return DecayFitResult(
                r_squared=r_squared,
                C0=C0,
                tau=tau,
                residuals=residuals,
                confidence_interval=ci,
                is_exponential=r_squared >= ANALYSIS['min_r_squared']
            )
        
        except Exception as e:
            print(f"Failed to fit exponential for {model_name}: {e}")
            return DecayFitResult(0, 0, 0, np.array([]), (0, 0), False)
    
    def test_discrete_vs_continuous(self, model_name: str) -> Dict:
        """
        Test whether decay is discrete (step functions) or continuous.
        
        Hypothesis: Discrete processing shows larger gradients at integer depths.
        Statistical test: Mann-Whitney U for gradient differences.
        """
        if not self.results:
            raise ValueError("No results loaded")
        
        model_data = self.results['by_model'].get(model_name, [])
        
        # Get scores by depth
        depth_scores = {}
        for trial in model_data:
            depth = trial['depth']
            score = trial['scores']['total']
            if depth not in depth_scores:
                depth_scores[depth] = []
            depth_scores[depth].append(score)
        
        depths = sorted(depth_scores.keys())
        means = [np.mean(depth_scores[d]) for d in depths]
        
        # Calculate gradients
        gradients = []
        gradient_types = []
        
        for i in range(len(depths) - 1):
            grad = (means[i+1] - means[i]) / (depths[i+1] - depths[i])
            gradients.append(abs(grad))  # Use absolute gradient
            
            # Classify gradient type
            if depths[i] % 1.0 == 0.5:  # Transition FROM fractional
                gradient_types.append('fractional_to_integer')
            elif depths[i] % 1.0 == 0:  # Transition FROM integer
                gradient_types.append('integer_to_fractional')
        
        # Separate gradients by type
        frac_to_int = [g for g, t in zip(gradients, gradient_types) 
                      if t == 'fractional_to_integer']
        int_to_frac = [g for g, t in zip(gradients, gradient_types) 
                      if t == 'integer_to_fractional']
        
        # Statistical test
        if frac_to_int and int_to_frac:
            statistic, p_value = mannwhitneyu(frac_to_int, int_to_frac, 
                                             alternative='two-sided')
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(frac_to_int) + np.var(int_to_frac)) / 2)
            if pooled_std > 0:
                cohens_d = abs(np.mean(frac_to_int) - np.mean(int_to_frac)) / pooled_std
            else:
                cohens_d = 0
        else:
            p_value = 1.0
            cohens_d = 0
        
        return {
            'model': model_name,
            'evidence_for_discrete': p_value < ANALYSIS['significance_level'],
            'p_value': p_value,
            'effect_size': cohens_d,
            'mean_gradient_frac_to_int': np.mean(frac_to_int) if frac_to_int else 0,
            'mean_gradient_int_to_frac': np.mean(int_to_frac) if int_to_frac else 0,
            'interpretation': 'DISCRETE' if p_value < 0.05 else 'CONTINUOUS'
        }
    
    def compare_models(self) -> Dict:
        """
        Compare decay patterns between models.
        
        Key comparison: Sampling sensitivity (R² change).
        """
        if not self.results:
            raise ValueError("No results loaded")
        
        comparison = {}
        
        for model in self.results['metadata']['models']:
            # Fit exponential
            fit = self.fit_exponential_decay(model)
            
            # Test discrete vs continuous
            discrete_test = self.test_discrete_vs_continuous(model)
            
            comparison[model] = {
                'exponential_fit': {
                    'r_squared': fit.r_squared,
                    'C0': fit.C0,
                    'tau': fit.tau,
                    'is_exponential': fit.is_exponential
                },
                'processing_type': discrete_test['interpretation'],
                'discrete_evidence': {
                    'p_value': discrete_test['p_value'],
                    'effect_size': discrete_test['effect_size']
                }
            }
        
        # Calculate divergence if both models present
        if len(comparison) >= 2:
            models = [k for k in comparison.keys() if k != 'divergence']
            
            # Calculate pairwise divergences
            divergences = []
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    r2_diff = abs(comparison[models[i]]['exponential_fit']['r_squared'] - 
                                 comparison[models[j]]['exponential_fit']['r_squared'])
                    divergences.append({
                        'pair': f"{models[i]} vs {models[j]}",
                        'r_squared_difference': r2_diff,
                        'significant': r2_diff > 0.3
                    })
            
            comparison['divergence'] = divergences
        
        return comparison
    
    def _bootstrap_r_squared(self, depth_scores: Dict, depths: np.ndarray, 
                            n_iterations: int = 1000) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for R².
        
        Based on Efron & Tibshirani (1993) bootstrap methods.
        """
        r_squared_values = []
        
        for _ in range(n_iterations):
            # Resample with replacement
            resampled_means = []
            for d in depths:
                scores = depth_scores[d]
                resampled = np.random.choice(scores, len(scores), replace=True)
                resampled_means.append(np.mean(resampled))
            
            # Fit exponential to resampled data
            try:
                def exponential_decay(d, C0, tau):
                    return C0 * np.exp(-d/tau)
                
                params, _ = curve_fit(
                    exponential_decay, 
                    depths, 
                    resampled_means,
                    p0=[0.6, 2.0],
                    maxfev=1000
                )
                
                predicted = exponential_decay(depths, *params)
                residuals = resampled_means - predicted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((resampled_means - np.mean(resampled_means))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r_squared_values.append(r2)
            except:
                continue
        
        # Calculate 95% CI
        if r_squared_values:
            ci_lower = np.percentile(r_squared_values, 2.5)
            ci_upper = np.percentile(r_squared_values, 97.5)
        else:
            ci_lower, ci_upper = 0, 0
        
        return (ci_lower, ci_upper)
    
    def plot_decay_curves(self, save_path: Optional[str] = None):
        """Generate visualization of decay curves."""
        if not self.results:
            raise ValueError("No results loaded")
        
        fig, axes = plt.subplots(1, len(self.results['metadata']['models']), 
                                 figsize=(7*len(self.results['metadata']['models']), 6))
        colors = {'gpt-3.5': '#1f77b4', 'claude-3-haiku': '#ff7f0e', 
                 'gemini-1.5-flash': '#2ca02c'}
        
        # Handle single model case
        if len(self.results['metadata']['models']) == 1:
            axes = [axes]
        
        for i, model in enumerate(self.results['metadata']['models']):
            ax = axes[i]
            
            # Get data
            model_data = self.results['by_model'][model]
            depth_scores = {}
            for trial in model_data:
                depth = trial['depth']
                score = trial['scores']['total']
                if depth not in depth_scores:
                    depth_scores[depth] = []
                depth_scores[depth].append(score)
            
            depths = np.array(sorted(depth_scores.keys()))
            means = np.array([np.mean(depth_scores[d]) for d in depths])
            stds = np.array([np.std(depth_scores[d]) for d in depths])
            
            # Plot data points with error bars
            ax.errorbar(depths, means, yerr=stds, 
                       fmt='o', capsize=5, capthick=2,
                       color=colors.get(model, 'black'),
                       label='Data')
            
            # Fit and plot exponential
            fit = self.fit_exponential_decay(model)
            if fit.C0 > 0:
                x_smooth = np.linspace(min(depths), max(depths), 100)
                y_smooth = fit.C0 * np.exp(-x_smooth/fit.tau)
                ax.plot(x_smooth, y_smooth, '--', 
                       color=colors.get(model, 'black'),
                       alpha=0.7, linewidth=2,
                       label=f'Exponential fit (R²={fit.r_squared:.3f})')
            
            ax.set_xlabel('Temporal Depth', fontsize=12)
            ax.set_ylabel('Coherence Score', fontsize=12)
            ax.set_title(f'{model.upper()}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle('Temporal Coherence Decay: Architectural Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report."""
        if not self.results:
            raise ValueError("No results loaded")
        
        comparison = self.compare_models()
        
        report = []
        report.append("=" * 60)
        report.append("LLM TIME DECAY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nExperiment: {self.results['metadata']['timestamp']}")
        report.append(f"Total trials: {self.results['metadata']['n_trials']}")
        report.append(f"Models: {', '.join(self.results['metadata']['models'])}")
        report.append(f"Depths tested: {self.results['metadata']['depths']}")
        
        report.append("\n" + "=" * 60)
        report.append("MODEL ANALYSIS")
        report.append("=" * 60)
        
        for model, analysis in comparison.items():
            if model == 'divergence':
                continue
            
            report.append(f"\n### {model.upper()} ###")
            
            exp = analysis['exponential_fit']
            report.append(f"\nExponential Fit:")
            report.append(f"  R²: {exp['r_squared']:.4f}")
            report.append(f"  Initial coherence (C₀): {exp['C0']:.4f}")
            report.append(f"  Decay constant (τ): {exp['tau']:.4f}")
            report.append(f"  Supports exponential: {exp['is_exponential']}")
            
            report.append(f"\nProcessing Type: {analysis['processing_type']}")
            disc = analysis['discrete_evidence']
            report.append(f"  p-value: {disc['p_value']:.4f}")
            report.append(f"  Effect size: {disc['effect_size']:.4f}")
        
        if 'divergence' in comparison:
            report.append("\n" + "=" * 60)
            report.append("ARCHITECTURAL COMPARISON")
            report.append("=" * 60)
            
            for div in comparison['divergence']:
                report.append(f"\n{div['pair']}:")
                report.append(f"  R² Divergence: {div['r_squared_difference']:.4f}")
                report.append(f"  Significant difference: {div['significant']}")
            
            # Check if any pair shows significant difference
            if any(d['significant'] for d in comparison['divergence']):
                report.append("\n*** MAJOR FINDING ***")
                report.append("Models show fundamentally different temporal processing!")
                report.append("This suggests distinct architectural mechanisms.")
        
        report.append("\n" + "=" * 60)
        report.append("INTERPRETATION")
        report.append("=" * 60)
        
        # Determine pattern
        models = [m for m in comparison.keys() if m != 'divergence']
        processing_types = {m: comparison[m]['processing_type'] for m in models}
        
        # Count processing types
        type_counts = {}
        for ptype in processing_types.values():
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        if len(type_counts) == 1:
            # All models same type
            ptype = list(type_counts.keys())[0]
            report.append(f"\nAll models show {ptype} processing")
            report.append("Suggests universal LLM temporal mechanism")
        else:
            # Different types
            report.append("\nProcessing type by model:")
            for model, ptype in processing_types.items():
                report.append(f"  {model}: {ptype}")
            report.append("\nARCHITECTURAL DIVERGENCE CONFIRMED!")
            report.append("Different models use different temporal mechanisms")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LLM time decay results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate analysis report')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer(args.results)
    
    # Run analysis
    print("\nComparing models...")
    comparison = analyzer.compare_models()
    
    # Print summary
    for model, analysis in comparison.items():
        if model != 'divergence':
            print(f"\n{model}:")
            print(f"  R²: {analysis['exponential_fit']['r_squared']:.3f}")
            print(f"  Type: {analysis['processing_type']}")
    
    if 'divergence' in comparison:
        print("\nDivergences:")
        for div in comparison['divergence']:
            print(f"  {div['pair']}: {div['r_squared_difference']:.3f}")
    
    # Generate plot if requested
    if args.plot:
        plot_path = Path(args.output_dir) / 'figures' / 'decay_curves.png'
        analyzer.plot_decay_curves(str(plot_path))
    
    # Generate report if requested
    if args.report:
        report_path = Path(args.output_dir) / 'tables' / 'analysis_report.txt'
        report = analyzer.generate_report(str(report_path))
        print("\n" + report)


if __name__ == "__main__":
    main()