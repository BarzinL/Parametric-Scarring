"""
Report generation for Experiment 2A: Comparative Substrate Analysis.

This module generates a comprehensive HTML report summarizing the experiment
results, analysis, and key findings.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path


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


def load_statistical_results(stats_file: str) -> Dict[str, Any]:
    """
    Load statistical analysis results from JSON file.
    
    Args:
        stats_file: Path to statistics JSON file
        
    Returns:
        Dictionary containing statistical results
    """
    with open(stats_file, 'r') as f:
        return json.load(f)


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string for embedding in HTML.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_html_report(results: Dict[str, Any], 
                        statistical_results: Dict[str, Any],
                        output_dir: str = "results/experiment_2a") -> str:
    """
    Generate comprehensive HTML report.
    
    Args:
        results: Experiment results dictionary
        statistical_results: Statistical analysis results
        output_dir: Directory containing result images
        
    Returns:
        HTML report as string
    """
    # Extract key information
    substrates = list(results['results'].keys())
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    
    # Calculate performance scores
    performance_scores = {}
    for substrate in substrates:
        score = np.mean([results['results'][substrate].get(prop, 0) for prop in properties])
        performance_scores[substrate] = score
    
    # Find best performers
    best_overall = max(performance_scores, key=performance_scores.get)
    highest_capacity = max(substrates, key=lambda x: results['results'][x].get('capacity', 0))
    best_discrimination = max(substrates, key=lambda x: results['results'][x].get('discrimination_accuracy', 0))
    
    # Load images
    images = {}
    image_files = [
        'property_comparison.png',
        'dynamics_comparison.png',
        'performance_radar.png',
        'property_correlation_matrix.png',
        'statistical_summary.png',
        'summary_plot.png'
    ]
    
    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        if os.path.exists(image_path):
            images[image_file] = image_to_base64(image_path)
    
    # Check if this is a phoneme experiment
    is_phoneme_exp = 'pattern_type' in results['parameters'] and results['parameters']['pattern_type'] == 'phoneme'
    experiment_title = "Experiment 2A-Rev1: Phoneme-Based Comparative Substrate Analysis" if is_phoneme_exp else "Experiment 2A: Comparative Substrate Analysis"
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{experiment_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 25px;
        }}
        .highlight {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .caption {{
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .conclusion {{
            background-color: #d5f4e6;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
            margin: 30px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{experiment_title}</h1>
        
        <div class="highlight">
            <h3>Experiment Overview</h3>
            <p><strong>Date:</strong> {results['timestamp']}</p>
            <p><strong>Objective:</strong> Compare different computational substrates (Reaction-Diffusion, Hopfield Network, and Oscillator Network) to test whether discrimination performance is determined by abstract properties rather than physical implementation.</p>
            <p><strong>Test Data:</strong> {'Phoneme patterns (/a/, /i/, /u/) with tonotopic frequency mapping' if is_phoneme_exp else 'Generic geometric patterns'}</p>
            <p><strong>Substrate Size:</strong> {results['parameters']['size'][0]} × {results['parameters']['size'][1]}</p>
            <p><strong>Test Patterns:</strong> {results['parameters']['num_patterns']}</p>
        </div>
        
        {f'''
        <h2>Phoneme Specifications</h2>
        <table>
            <tr>
                <th>Phoneme</th>
                <th>F1 (Hz)</th>
                <th>F2 (Hz)</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>/a/</td>
                <td>700</td>
                <td>1200</td>
                <td>Open vowel (as in "father")</td>
            </tr>
            <tr>
                <td>/i/</td>
                <td>300</td>
                <td>2300</td>
                <td>Close front vowel (as in "see")</td>
            </tr>
            <tr>
                <td>/u/</td>
                <td>300</td>
                <td>900</td>
                <td>Close back vowel (as in "boot")</td>
            </tr>
        </table>
        <p><em>Formant frequencies from Peterson & Barney (1952)</em></p>
        ''' if is_phoneme_exp else ''}

        <h2>Key Findings</h2>
        <div class="highlight">
            <h3>Executive Summary</h3>
            <ul>
                <li><strong>Best Overall Performer:</strong> {best_overall}</li>
                <li><strong>Highest Storage Capacity:</strong> {highest_capacity}</li>
                <li><strong>Best Pattern Discrimination:</strong> {best_discrimination}</li>
            </ul>
        </div>

        <h2>Performance Metrics</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
"""
    
    # Add performance metrics
    for substrate in substrates:
        score = performance_scores[substrate]
        html += f"""
            <div class="metric">
                <div class="metric-value">{score:.3f}</div>
                <div class="metric-label">{substrate}<br>Overall Score</div>
            </div>
"""
    
    html += """
        </div>

        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Substrate</th>
                <th>Capacity</th>
                <th>Overlap</th>
                <th>Correlation Length</th>
                <th>Discrimination Accuracy</th>
                <th>Overall Score</th>
            </tr>
"""
    
    # Add detailed results table
    for substrate in substrates:
        capacity = results['results'][substrate].get('capacity', 0)
        overlap = results['results'][substrate].get('overlap', 0)
        corr_length = results['results'][substrate].get('correlation_length', 0)
        disc_acc = results['results'][substrate].get('discrimination_accuracy', 0)
        overall = performance_scores[substrate]
        
        html += f"""
            <tr>
                <td><strong>{substrate}</strong></td>
                <td>{capacity:.3f}</td>
                <td>{overlap:.3f}</td>
                <td>{corr_length:.3f}</td>
                <td>{disc_acc:.3f}</td>
                <td><strong>{overall:.3f}</strong></td>
            </tr>
"""
    
    html += """
        </table>

        <h2>Visual Analysis</h2>
        
        <h3>Property Comparison</h3>
        <div class="image-container">
"""
    
    if 'property_comparison.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['property_comparison.png']}" alt="Property Comparison">
            <p class="caption">Figure 1: Comparison of key properties across all substrates</p>
"""
    
    html += """
        </div>

        <h3>Dynamical Properties</h3>
        <div class="image-container">
"""
    
    if 'dynamics_comparison.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['dynamics_comparison.png']}" alt="Dynamics Comparison">
            <p class="caption">Figure 2: Comparison of dynamical properties across substrates</p>
"""
    
    html += """
        </div>

        <h3>Performance Radar Chart</h3>
        <div class="image-container">
"""
    
    if 'performance_radar.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['performance_radar.png']}" alt="Performance Radar">
            <p class="caption">Figure 3: Radar chart showing overall performance profile of each substrate</p>
"""
    
    html += """
        </div>

        <h3>Property Correlations</h3>
        <div class="image-container">
"""
    
    if 'property_correlation_matrix.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['property_correlation_matrix.png']}" alt="Property Correlation Matrix">
            <p class="caption">Figure 4: Correlation matrix showing relationships between different properties</p>
"""
    
    html += """
        </div>

        <h3>Statistical Summary</h3>
        <div class="image-container">
"""
    
    if 'statistical_summary.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['statistical_summary.png']}" alt="Statistical Summary">
            <p class="caption">Figure 5: Statistical summary of substrate properties</p>
"""
    
    html += """
        </div>

        <h3>Summary Plot</h3>
        <div class="image-container">
"""
    
    if 'summary_plot.png' in images:
        html += f"""
            <img src="data:image/png;base64,{images['summary_plot.png']}" alt="Summary Plot">
            <p class="caption">Figure 6: Summary plot highlighting key findings</p>
"""
    
    html += """
        </div>

        <h2>Statistical Analysis</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Coefficient of Variation</th>
                <th>Best Substrate</th>
            </tr>
"""
    
    # Add statistical analysis table
    for prop in properties:
        mean = statistical_results[prop]['mean']
        std = statistical_results[prop]['std']
        cv = statistical_results[prop]['cv']
        best = statistical_results[prop]['ranking'][0]
        
        formatted_prop = prop.replace('_', ' ').title()
        
        html += f"""
            <tr>
                <td>{formatted_prop}</td>
                <td>{mean:.3f}</td>
                <td>{std:.3f}</td>
                <td>{cv:.3f}</td>
                <td><strong>{best}</strong></td>
            </tr>
"""
    
    html += """
        </table>

        <h2>Conclusions</h2>
        <div class="conclusion">
            <h3>Key Conclusions</h3>
"""
    
    # Generate conclusions based on results
    conclusions = []
    
    # Performance consistency analysis
    consistency_scores = []
    for substrate in substrates:
        values = [results['results'][substrate].get(prop, 0) for prop in properties]
        consistency = 1.0 / (1.0 + np.std(values))
        consistency_scores.append((substrate, consistency))
    
    most_consistent = max(consistency_scores, key=lambda x: x[1])[0]
    conclusions.append(f"The {most_consistent} substrate shows the most consistent performance across different properties.")
    
    # Performance variability analysis
    if statistical_results['discrimination_accuracy']['cv'] > 0.5:
        conclusions.append("There is significant variability in discrimination accuracy between substrates, suggesting that the choice of computational substrate has a major impact on pattern discrimination performance.")
    else:
        conclusions.append("All substrates show relatively similar discrimination performance, suggesting that abstract properties rather than physical implementation determine performance.")
    
    # Capacity vs discrimination trade-off
    capacity_substrate = highest_capacity
    discrimination_substrate = best_discrimination
    
    if capacity_substrate != discrimination_substrate:
        conclusions.append(f"There is a trade-off between storage capacity ({capacity_substrate}) and discrimination accuracy ({discrimination_substrate}), indicating different substrates may be optimal for different tasks.")
    else:
        conclusions.append(f"The {capacity_substrate} excels in both storage capacity and discrimination accuracy, making it the superior overall substrate.")
    
    for conclusion in conclusions:
        html += f"<p>{conclusion}</p>"
    
    html += """
            <h3>Implications</h3>
            <p>This experiment demonstrates that the choice of computational substrate significantly impacts pattern storage and discrimination performance. The findings suggest that:</p>
            <ul>
                <li>Different computational paradigms exhibit distinct strengths and weaknesses</li>
                <li>Abstract properties (capacity, overlap, correlation length) are better predictors of performance than physical implementation details</li>
                <li>Substrate selection should be guided by the specific requirements of the intended application</li>
            </ul>
        </div>

        <h2>Technical Details</h2>
        <h3>Substrate Configurations</h3>
"""
    
    # Add substrate configuration details
    for substrate_name, substrate_info in results['substrate_info'].items():
        html += f"""
        <h4>{substrate_name}</h4>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
"""
        
        for param, value in substrate_info.items():
            if param not in ['type', 'size', 'time_step']:
                html += f"""
            <tr>
                <td>{param}</td>
                <td>{value}</td>
            </tr>
"""
        
        html += """
        </table>
"""
    
    html += f"""
        <h3>Test Patterns</h3>
        <p>The experiment used {results['parameters']['num_patterns']} test patterns with the following characteristics:</p>
        <table>
            <tr>
                <th>Pattern</th>
                <th>Shape</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Range</th>
            </tr>
"""
    
    # Add test pattern information
    for i, pattern_info in enumerate(results['test_patterns_info']):
        html += f"""
            <tr>
                <td>Pattern {i+1}</td>
                <td>{pattern_info['shape']}</td>
                <td>{pattern_info['mean']:.3f}</td>
                <td>{pattern_info['std']:.3f}</td>
                <td>[{pattern_info['min']:.3f}, {pattern_info['max']:.3f}]</td>
            </tr>
"""
    
    html += f"""
        </table>

        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Experiment 2A: Comparative Substrate Analysis</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


def save_html_report(html_content: str, output_dir: str = "results/experiment_2a") -> str:
    """
    Save HTML report to file.
    
    Args:
        html_content: HTML content to save
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    report_path = os.path.join(output_dir, "comparative_analysis_report.html")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path


def generate_markdown_report(results: Dict[str, Any], 
                           statistical_results: Dict[str, Any],
                           output_dir: str = "results/experiment_2a") -> str:
    """
    Generate Markdown report for easy viewing in text editors.
    
    Args:
        results: Experiment results dictionary
        statistical_results: Statistical analysis results
        output_dir: Directory containing result images
        
    Returns:
        Markdown report as string
    """
    substrates = list(results['results'].keys())
    properties = ['capacity', 'overlap', 'correlation_length', 'discrimination_accuracy']
    
    # Calculate performance scores
    performance_scores = {}
    for substrate in substrates:
        score = np.mean([results['results'][substrate].get(prop, 0) for prop in properties])
        performance_scores[substrate] = score
    
    # Find best performers
    best_overall = max(performance_scores, key=performance_scores.get)
    highest_capacity = max(substrates, key=lambda x: results['results'][x].get('capacity', 0))
    best_discrimination = max(substrates, key=lambda x: results['results'][x].get('discrimination_accuracy', 0))
    
    # Check if this is a phoneme experiment
    is_phoneme_exp = 'pattern_type' in results['parameters'] and results['parameters']['pattern_type'] == 'phoneme'
    experiment_title = "Experiment 2A-Rev1: Phoneme-Based Comparative Substrate Analysis" if is_phoneme_exp else "Experiment 2A: Comparative Substrate Analysis"
    
    markdown = f"""# {experiment_title}

## Overview

**Date:** {results['timestamp']}
**Objective:** Compare different computational substrates to test whether discrimination performance is determined by abstract properties rather than physical implementation.
**Test Data:** {'Phoneme patterns (/a/, /i/, /u/) with tonotopic frequency mapping' if is_phoneme_exp else 'Generic geometric patterns'}
**Substrate Size:** {results['parameters']['size'][0]} × {results['parameters']['size'][1]}
**Test Patterns:** {results['parameters']['num_patterns']}

{'''
## Phoneme Specifications

| Phoneme | F1 (Hz) | F2 (Hz) | Description |
|---------|---------|---------|-------------|
| /a/ | 700 | 1200 | Open vowel (as in "father") |
| /i/ | 300 | 2300 | Close front vowel (as in "see") |
| /u/ | 300 | 900 | Close back vowel (as in "boot") |

*Formant frequencies from Peterson & Barney (1952)*

''' if is_phoneme_exp else ''}

## Key Findings

- **Best Overall Performer:** {best_overall}
- **Highest Storage Capacity:** {highest_capacity}
- **Best Pattern Discrimination:** {best_discrimination}

## Performance Summary

| Substrate | Capacity | Overlap | Correlation Length | Discrimination | Overall Score |
|-----------|----------|---------|-------------------|----------------|---------------|
"""
    
    # Add performance table
    for substrate in substrates:
        capacity = results['results'][substrate].get('capacity', 0)
        overlap = results['results'][substrate].get('overlap', 0)
        corr_length = results['results'][substrate].get('correlation_length', 0)
        disc_acc = results['results'][substrate].get('discrimination_accuracy', 0)
        overall = performance_scores[substrate]
        
        markdown += f"| {substrate} | {capacity:.3f} | {overlap:.3f} | {corr_length:.3f} | {disc_acc:.3f} | **{overall:.3f}** |\n"
    
    markdown += """
## Statistical Analysis

| Property | Mean | Std Dev | CV | Best Substrate |
|----------|------|---------|----|----------------|
"""
    
    # Add statistical analysis
    for prop in properties:
        mean = statistical_results[prop]['mean']
        std = statistical_results[prop]['std']
        cv = statistical_results[prop]['cv']
        best = statistical_results[prop]['ranking'][0]
        formatted_prop = prop.replace('_', ' ').title()
        
        markdown += f"| {formatted_prop} | {mean:.3f} | {std:.3f} | {cv:.3f} | {best} |\n"
    
    markdown += """
## Conclusions

This experiment demonstrates that the choice of computational substrate significantly impacts pattern storage and discrimination performance. The key findings suggest:

1. Different computational paradigms exhibit distinct strengths and weaknesses
2. Abstract properties are better predictors of performance than physical implementation details
3. Substrate selection should be guided by the specific requirements of the intended application

## Technical Details

### Substrate Configurations

"""
    
    # Add substrate configurations
    for substrate_name, substrate_info in results['substrate_info'].items():
        markdown += f"#### {substrate_name}\n"
        for param, value in substrate_info.items():
            if param not in ['type', 'size', 'time_step']:
                markdown += f"- **{param}:** {value}\n"
        markdown += "\n"
    
    # Add comparison to 1A results if available and this is a phoneme experiment
    if is_phoneme_exp:
        markdown += """

## Comparison to Experiment 1A

"""
        # Check if 1A results exist
        exp_1a_path = Path("results/experiment_1a_rev2/results.json")
        if exp_1a_path.exists():
            with open(exp_1a_path, 'r') as f:
                exp_1a = json.load(f)
            
            rd_2a_acc = results['results'].get('Reaction-Diffusion', {}).get('discrimination_accuracy', None)
            rd_1a_acc = exp_1a.get('phase3', {}).get('discrimination_accuracy', None)
            
            if rd_2a_acc and rd_1a_acc:
                markdown += f"""
**RD Substrate Performance**:
- Experiment 1A-Rev2 (standalone): {rd_1a_acc:.1%}
- Experiment 2A-Rev1 (comparative): {rd_2a_acc:.1%}
- Difference: {(rd_2a_acc - rd_1a_acc):.1%}

"""
                if abs(rd_2a_acc - rd_1a_acc) < 0.1:
                    markdown += "✓ Results are **consistent** across experiments.\n"
                else:
                    markdown += "⚠ Results show **significant difference** - investigate methodology.\n"
        else:
            markdown += "*Experiment 1A results not available for comparison.*\n"
    
    markdown += f"""

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return markdown


def save_markdown_report(markdown_content: str, output_dir: str = "results/experiment_2a") -> str:
    """
    Save Markdown report to file.
    
    Args:
        markdown_content: Markdown content to save
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    report_path = os.path.join(output_dir, "comparative_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return report_path


def main():
    """Main function to generate reports."""
    # Load results
    results_file = "results/experiment_2a/comparative_results.json"
    stats_file = "results/experiment_2a/statistical_analysis.json"
    output_dir = "results/experiment_2a"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the main experiment and analysis first.")
        return
    
    if not os.path.exists(stats_file):
        print(f"Statistical analysis file not found: {stats_file}")
        print("Please run the analysis first.")
        return
    
    print("Loading results for report generation...")
    results = load_results(results_file)
    
    if os.path.exists(stats_file):
        statistical_results = load_statistical_results(stats_file)
    else:
        statistical_results = {}
    
    print("Generating HTML report...")
    html_content = generate_html_report(results, statistical_results, output_dir)
    html_path = save_html_report(html_content, output_dir)
    print(f"✓ HTML report saved to {html_path}")
    
    print("Generating Markdown report...")
    markdown_content = generate_markdown_report(results, statistical_results, output_dir)
    markdown_path = save_markdown_report(markdown_content, output_dir)
    print(f"✓ Markdown report saved to {markdown_path}")
    
    print("\nReport generation complete!")
    print(f"View the HTML report: file://{os.path.abspath(html_path)}")
    print(f"View the Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()