"""
Analysis and visualization for Experiment 2A: Comparative Substrate Analysis.

This module provides functions to analyze and visualize the results
from the comparative substrate experiment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import pandas as pd
from scipy import stats


def load_results(results_file: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        results_file: Path to results JSON file
        
    Returns:
        Dictionary containing experiment results
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def create_property_comparison(results: Dict[str, Any], 
                             output_dir: str = "results/experiment_2a") -> None:
    """
    Create comparison plots for substrate properties.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    # Extract property data
    substrates = list(results['results'].keys())
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    
    # Create data for plotting
    data = []
    for substrate in substrates:
        for prop in properties:
            value = results['results'][substrate].get(prop, 0)
            data.append({
                'Substrate': substrate,
                'Property': prop.replace('_', ' ').title(),
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Property', y='Value', hue='Substrate')
    plt.title('Substrate Property Comparison', fontsize=16)
    plt.xlabel('Property', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Substrate')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/property_comparison.png", dpi=300)
    plt.close()
    
    # Create individual property plots
    for prop in properties:
        plt.figure(figsize=(8, 6))
        values = [results['results'][substrate].get(prop, 0) for substrate in substrates]
        
        bars = plt.bar(substrates, values)
        plt.title(f'{prop.replace("_", " ").title()} Comparison', fontsize=14)
        plt.ylabel('Value', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prop}_comparison.png", dpi=300)
        plt.close()


def create_dynamics_comparison(results: Dict[str, Any],
                             output_dir: str = "results/experiment_2a") -> None:
    """
    Create comparison plots for dynamical properties.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    # Extract dynamics data
    substrates = list(results['results'].keys())
    dynamics_props = ['convergence_rate', 'stability', 'dominant_frequency']
    
    # Create data for plotting
    data = []
    for substrate in substrates:
        for prop in dynamics_props:
            value = results['results'][substrate].get(prop, 0)
            # Convert string to float
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
            
            # Handle very small values
            if value < 1e-15:
                value = 1e-15
            
            data.append({
                'Substrate': substrate,
                'Property': prop.replace('_', ' ').title(),
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Check if we need log scale (values span > 2 orders of magnitude)
    value_range = df['Value'].max() / (df['Value'].min() + 1e-15)
    use_log_scale = value_range > 100
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Property', y='Value', hue='Substrate')
    
    if use_log_scale:
        ax.set_yscale('log')
        plt.ylabel('Value (log scale)', fontsize=12)
    else:
        plt.ylabel('Value', fontsize=12)
    
    plt.title('Dynamical Properties Comparison', fontsize=16)
    plt.xlabel('Property', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Substrate')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dynamics_comparison.png", dpi=300)
    plt.close()


def create_performance_radar(results: Dict[str, Any],
                           output_dir: str = "results/experiment_2a") -> None:
    """
    Create radar chart for overall performance comparison.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    from math import pi
    
    # Extract substrates and properties
    substrates = list(results['results'].keys())
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    
    # Number of properties
    N = len(properties)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for each substrate
    colors = plt.cm.Set3(np.linspace(0, 1, len(substrates)))
    
    # Plot each substrate
    for i, substrate in enumerate(substrates):
        values = []
        for prop in properties:
            # Normalize values to [0, 1] for radar chart
            value = results['results'][substrate].get(prop, 0)
            values.append(value)
        
        values += values[:1]  # Complete the loop
        
        # Plot values
        ax.plot(angles, values, 'o-', linewidth=2, label=substrate, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add property labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([prop.replace('_', ' ').title() for prop in properties])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    # Add title
    plt.title('Substrate Performance Radar Chart', size=16, y=1.08)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_radar.png", dpi=300)
    plt.close()


def create_correlation_matrix(results: Dict[str, Any],
                            output_dir: str = "results/experiment_2a") -> None:
    """
    Create correlation matrix of properties across substrates.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    # Extract all numeric properties
    substrates = list(results['results'].keys())
    all_properties = set()
    
    for substrate in substrates:
        all_properties.update(results['results'][substrate].keys())
    
    # Filter numeric properties
    numeric_properties = []
    for prop in all_properties:
        try:
            # Check if all values are numeric
            if all(isinstance(results['results'][substrate].get(prop, 0), (int, float)) 
                   for substrate in substrates):
                numeric_properties.append(prop)
        except:
            continue
    
    # Create correlation matrix
    property_matrix = []
    for substrate in substrates:
        row = [results['results'][substrate].get(prop, 0) for prop in numeric_properties]
        property_matrix.append(row)
    
    property_df = pd.DataFrame(property_matrix, 
                              index=substrates, 
                              columns=numeric_properties)
    
    # Compute correlation matrix
    corr_matrix = property_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Property Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/property_correlation_matrix.png", dpi=300)
    plt.close()


def analyze_statistical_significance(results: Dict[str, Any],
                                   output_dir: str = "results/experiment_2a") -> Dict[str, Any]:
    """
    Perform statistical analysis of substrate differences.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save analysis
        
    Returns:
        Dictionary containing statistical analysis results
    """
    substrates = list(results['results'].keys())
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    
    statistical_results = {}
    
    for prop in properties:
        values = [results['results'][substrate].get(prop, 0) for substrate in substrates]
        
        # Basic statistics
        statistical_results[prop] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values)
        }
        
        # Coefficient of variation
        if statistical_results[prop]['mean'] != 0:
            statistical_results[prop]['cv'] = statistical_results[prop]['std'] / statistical_results[prop]['mean']
        else:
            statistical_results[prop]['cv'] = 0
        
        # Rank substrates by this property
        ranked_substrates = sorted(substrates, 
                                 key=lambda x: results['results'][x].get(prop, 0), 
                                 reverse=True)
        statistical_results[prop]['ranking'] = ranked_substrates
    
    # Create summary table
    summary_data = []
    for prop in properties:
        row = [
            prop.replace('_', ' ').title(),
            f"{statistical_results[prop]['mean']:.3f}",
            f"{statistical_results[prop]['std']:.3f}",
            f"{statistical_results[prop]['cv']:.3f}",
            statistical_results[prop]['ranking'][0]  # Best substrate
        ]
        summary_data.append(row)
    
    # Create summary table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Property', 'Mean', 'Std', 'CV', 'Best Substrate'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Statistical Summary of Substrate Properties', fontsize=16)
    plt.savefig(f"{output_dir}/statistical_summary.png", dpi=300)
    plt.close()
    
    # Save statistical results
    with open(f"{output_dir}/statistical_analysis.json", 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    
    return statistical_results


def create_comprehensive_report(results: Dict[str, Any],
                              output_dir: str = "results/experiment_2a") -> None:
    """
    Create comprehensive visual report of the experiment.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    print("Creating comprehensive visual report...")
    
    # Create all visualizations
    create_property_comparison(results, output_dir)
    print("✓ Property comparison plots created")
    
    create_dynamics_comparison(results, output_dir)
    print("✓ Dynamics comparison plots created")
    
    create_performance_radar(results, output_dir)
    print("✓ Performance radar chart created")
    
    create_correlation_matrix(results, output_dir)
    print("✓ Property correlation matrix created")
    
    statistical_results = analyze_statistical_significance(results, output_dir)
    print("✓ Statistical analysis completed")
    
    # Create summary plot
    create_summary_plot(results, statistical_results, output_dir)
    print("✓ Summary plot created")
    
    print(f"Visual report saved to {output_dir}")


def create_summary_plot(results: Dict[str, Any], 
                       statistical_results: Dict[str, Any],
                       output_dir: str = "results/experiment_2a") -> None:
    """
    Create summary plot highlighting key findings.
    
    Args:
        results: Experiment results dictionary
        statistical_results: Statistical analysis results
        output_dir: Directory to save plot
    """
    substrates = list(results['results'].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 2A: Comparative Substrate Analysis Summary', fontsize=20)
    
    # Plot 1: Overall performance score
    ax1 = axes[0, 0]
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    performance_scores = []
    
    for substrate in substrates:
        score = np.mean([results['results'][substrate].get(prop, 0) for prop in properties])
        performance_scores.append(score)
    
    bars = ax1.bar(substrates, performance_scores)
    ax1.set_title('Overall Performance Score', fontsize=14)
    ax1.set_ylabel('Normalized Score', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, performance_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Property variability
    ax2 = axes[0, 1]
    prop_variability = [statistical_results[prop]['cv'] for prop in properties]
    
    bars = ax2.bar(range(len(properties)), prop_variability)
    ax2.set_title('Property Variability (CV)', fontsize=14)
    ax2.set_ylabel('Coefficient of Variation', fontsize=12)
    ax2.set_xticks(range(len(properties)))
    ax2.set_xticklabels([prop.replace('_', ' ').title() for prop in properties], rotation=45)
    
    # Plot 3: Best substrate per property
    ax3 = axes[1, 0]
    best_substrates = [statistical_results[prop]['ranking'][0] for prop in properties]
    substrate_counts = {substrate: best_substrates.count(substrate) for substrate in substrates}
    
    bars = ax3.bar(substrate_counts.keys(), substrate_counts.values())
    ax3.set_title('Number of "Best" Properties per Substrate', fontsize=14)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Key insights text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Generate key insights
    insights = []
    
    # Best overall performer
    best_overall = substrates[np.argmax(performance_scores)]
    insights.append(f"Best Overall: {best_overall}")
    
    # Most consistent performer
    consistency_scores = []
    for substrate in substrates:
        values = [results['results'][substrate].get(prop, 0) for prop in properties]
        consistency = 1.0 / (1.0 + np.std(values))  # Lower std = more consistent
        consistency_scores.append(consistency)
    
    most_consistent = substrates[np.argmax(consistency_scores)]
    insights.append(f"Most Consistent: {most_consistent}")
    
    # Highest capacity
    highest_capacity = max(substrates, key=lambda x: results['results'][x].get('capacity', 0))
    insights.append(f"Highest Capacity: {highest_capacity}")
    
    # Best discrimination
    best_discrimination = max(substrates, key=lambda x: results['results'][x].get('discrimination_accuracy', 0))
    insights.append(f"Best Discrimination: {best_discrimination}")
    
    # Display insights
    insight_text = "Key Findings:\n\n" + "\n".join([f"• {insight}" for insight in insights])
    ax4.text(0.1, 0.5, insight_text, fontsize=14, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_plot.png", dpi=300)
    plt.close()


def add_phoneme_specific_analysis(results: Dict[str, Any],
                                 output_dir: str = "results/experiment_2a") -> None:
    """
    Add phoneme-specific analysis to the results.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("PHONEME-SPECIFIC ANALYSIS")
    print("="*70 + "\n")
    
    # Check if this is a phoneme experiment
    if 'pattern_type' not in results['parameters'] or results['parameters']['pattern_type'] != 'phoneme':
        print("Not a phoneme experiment - skipping phoneme-specific analysis")
        return
    
    phonemes = results['parameters'].get('phonemes', ['a', 'i', 'u'])
    
    # Analyze which phonemes are hardest to discriminate
    for substrate_name in results['results']:
        if 'discrimination' in results['results'][substrate_name] and 'matrix' in results['results'][substrate_name]['discrimination']:
            disc_matrix = results['results'][substrate_name]['discrimination']['matrix']
            
            print(f"\n{substrate_name}:")
            for phoneme in phonemes:
                if phoneme in disc_matrix:
                    result = disc_matrix[phoneme]
                    sims = result['similarities']
                    correct = result['correct']
                    
                    # Find most confused phoneme
                    other_phonemes = [p for p in phonemes if p != phoneme]
                    max_confusion = max(sims[p] for p in other_phonemes)
                    confused_with = max(other_phonemes, key=lambda p: sims[p])
                    
                    status = "✓" if correct else f"✗ (→ /{confused_with}/)"
                    print(f"  /{phoneme}/: {status} (max confusion: {max_confusion:.3f} with /{confused_with}/)")
    
    # Create phoneme confusion matrix visualization
    create_phoneme_confusion_matrices(results, output_dir)


def create_phoneme_confusion_matrices(results: Dict[str, Any],
                                    output_dir: str = "results/experiment_2a") -> None:
    """
    Create confusion matrices for phoneme discrimination.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
    """
    # Check if this is a phoneme experiment
    if 'pattern_type' not in results['parameters'] or results['parameters']['pattern_type'] != 'phoneme':
        return
    
    phonemes = results['parameters'].get('phonemes', ['a', 'i', 'u'])
    substrates = list(results['results'].keys())
    
    # Create figure with subplots for each substrate
    fig, axes = plt.subplots(1, len(substrates), figsize=(5*len(substrates), 5))
    if len(substrates) == 1:
        axes = [axes]
    
    for i, substrate_name in enumerate(substrates):
        ax = axes[i]
        
        # Create confusion matrix
        confusion_matrix = np.zeros((len(phonemes), len(phonemes)))
        
        if 'discrimination' in results['results'][substrate_name] and 'matrix' in results['results'][substrate_name]['discrimination']:
            disc_matrix = results['results'][substrate_name]['discrimination']['matrix']
            
            for j, target_phoneme in enumerate(phonemes):
                if target_phoneme in disc_matrix:
                    result = disc_matrix[target_phoneme]
                    if result['correct']:
                        confusion_matrix[j, j] = 1  # Correct classification
                    else:
                        # Find confused phoneme
                        similarities = result['similarities']
                        other_phonemes = [p for p in phonemes if p != target_phoneme]
                        confused_with = max(other_phonemes, key=lambda p: similarities[p])
                        k = phonemes.index(confused_with)
                        confusion_matrix[j, k] = similarities[confused_with]
        
        # Plot heatmap
        im = ax.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(phonemes)))
        ax.set_yticks(range(len(phonemes)))
        ax.set_xticklabels([f'/{p}/' for p in phonemes])
        ax.set_yticklabels([f'/{p}/' for p in phonemes])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Target')
        ax.set_title(f'{substrate_name} Phoneme Confusion')
        
        # Add text annotations
        for j in range(len(phonemes)):
            for k in range(len(phonemes)):
                text = ax.text(k, j, f'{confusion_matrix[j, k]:.2f}',
                             ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phoneme_confusion_matrices.png", dpi=300)
    plt.close()


def main():
    """Main function to analyze results."""
    # Load results
    results_file = "results/experiment_2a/comparative_results.json"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the main experiment first.")
        return
    
    print("Loading experiment results...")
    results = load_results(results_file)
    
    # Create comprehensive report
    create_comprehensive_report(results)
    
    # Add phoneme-specific analysis if applicable
    add_phoneme_specific_analysis(results)
    
    print("\nAnalysis complete!")
    print("Check the results/experiment_2a directory for all visualization files.")


if __name__ == "__main__":
    main()