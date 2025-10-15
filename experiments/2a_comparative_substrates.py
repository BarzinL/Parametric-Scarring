"""
Experiment 2A: Comparative Substrate Analysis

Main experiment to compare different computational substrates (Reaction-Diffusion,
Hopfield Network, and Oscillator Network) for pattern storage and discrimination.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import substrates
from core.substrates.rd_substrate import RDSubstrate
from core.substrates.hopfield_substrate import HopfieldSubstrate
from core.substrates.oscillator_substrate import OscillatorSubstrate

# Import analysis tools
from core.analysis.properties import compare_substrates, measure_substrate_properties

# Import pattern generation
from core.patterns import generate_phoneme_patterns, generate_random_patterns


def create_test_patterns(size: Tuple[int, int], num_patterns: int = 5) -> List[np.ndarray]:
    """
    Create test patterns for the experiment.
    NOW USING PHONEME PATTERNS ONLY.
    
    Args:
        size: Size of patterns (height, width)
        num_patterns: Number of patterns to generate
        
    Returns:
        List of test patterns
    """
    # Generate phoneme patterns using existing function
    phoneme_patterns = generate_phoneme_patterns(size)
    
    # Should return 3 patterns for /a/, /i/, /u/
    return phoneme_patterns[:num_patterns]


def setup_substrates(size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Set up all substrate instances for comparison.
    
    Args:
        size: Size of substrates (height, width)
        
    Returns:
        Dictionary of substrate instances
    """
    substrates = {}
    
    # Reaction-Diffusion substrate
    substrates['Reaction-Diffusion'] = RDSubstrate(
        size=size,
        default_f=0.037,
        default_k=0.060,
        Du=0.16,
        Dv=0.08
    )
    
    # Hopfield Network substrate
    substrates['Hopfield Network'] = HopfieldSubstrate(
        size=size,
        temperature=0.5,
        dt=0.1
    )
    
    # Oscillator Network substrate
    substrates['Oscillator Network'] = OscillatorSubstrate(
        size=size,
        coupling_strength=0.1,
        natural_freq_range=(0.8, 1.2),
        dt=0.01
    )
    
    return substrates


def run_comparative_experiment(size: Tuple[int, int] = (128, 128),
                              num_patterns: int = 5,
                              output_dir: str = "results/experiment_2a") -> Dict[str, Any]:
    """
    Run the comparative substrate analysis experiment.
    
    Args:
        size: Size of substrates (height, width)
        num_patterns: Number of test patterns
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    print("Starting Experiment 2A-Rev1: Phoneme-Based Comparative Substrate Analysis")
    print(f"Substrate size: {size}")
    print(f"Number of patterns: {num_patterns}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test patterns
    print("\nGenerating test patterns...")
    test_patterns = create_test_patterns(size, num_patterns)
    
    # Define phoneme labels
    PHONEME_LABELS = ['a', 'i', 'u']
    labels = PHONEME_LABELS[:len(test_patterns)]
    
    # Save test patterns with phoneme labels
    for i, (pattern, label) in enumerate(zip(test_patterns, labels)):
        plt.figure(figsize=(6, 6))
        plt.imshow(pattern, cmap='viridis')
        plt.title(f"Phoneme /{label}/ Pattern")
        plt.colorbar()
        plt.savefig(f"{output_dir}/test_pattern_{i+1}.png")
        plt.close()
    
    # Set up substrates
    print("\nSetting up substrates...")
    substrates = setup_substrates(size)
    
    # Run comparison
    print("\nRunning comparative analysis...")
    results = compare_substrates(substrates, test_patterns)
    
    # Add substrate information
    substrate_info = {}
    for name, substrate in substrates.items():
        substrate_info[name] = substrate.get_info()
    
    # Compile experiment results
    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'size': size,
            'num_patterns': num_patterns,
            'substrate_types': list(substrates.keys()),
            'pattern_type': 'phoneme',  # NEW
            'phonemes': PHONEME_LABELS[:num_patterns]  # NEW
        },
        'substrate_info': substrate_info,
        'results': results,
        'test_patterns_info': [
            {
                'shape': pattern.shape,
                'mean': float(np.mean(pattern)),
                'std': float(np.std(pattern)),
                'min': float(np.min(pattern)),
                'max': float(np.max(pattern)),
                'label': label  # NEW
            }
            for pattern, label in zip(test_patterns, labels)
        ]
    }
    
    # Save results
    results_file = f"{output_dir}/comparative_results.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    return experiment_results


def test_individual_substrate(substrate_name: str, size: Tuple[int, int] = (128, 128),
                            output_dir: str = "results/experiment_2a") -> Dict[str, Any]:
    """
    Test an individual substrate for debugging and validation.
    
    Args:
        substrate_name: Name of substrate to test
        size: Size of substrate (height, width)
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing test results
    """
    print(f"\nTesting {substrate_name} substrate...")
    
    # Create substrate
    substrates = setup_substrates(size)
    substrate = substrates[substrate_name]
    
    # Create test patterns
    test_patterns = create_test_patterns(size, 3)
    
    # Test basic functionality
    results = {
        'substrate_info': substrate.get_info(),
        'basic_tests': {}
    }
    
    # Test initialization
    substrate.initialize()
    initial_state = substrate.get_state()
    results['basic_tests']['initialization'] = {
        'state_shape': initial_state.shape,
        'state_mean': float(np.mean(initial_state)),
        'state_std': float(np.std(initial_state))
    }
    
    # Test pattern injection
    substrate.inject_pattern(test_patterns[0])
    after_injection = substrate.get_state()
    results['basic_tests']['injection'] = {
        'state_mean': float(np.mean(after_injection)),
        'state_std': float(np.std(after_injection)),
        'injection_effect': float(np.mean(np.abs(after_injection - initial_state)))
    }
    
    # Test evolution
    substrate.evolve(steps=50)
    after_evolution = substrate.get_state()
    results['basic_tests']['evolution'] = {
        'state_mean': float(np.mean(after_evolution)),
        'state_std': float(np.std(after_evolution)),
        'evolution_effect': float(np.mean(np.abs(after_evolution - after_injection)))
    }
    
    # Test pattern storage
    substrate.reset()
    for i, pattern in enumerate(test_patterns):
        substrate.store_pattern(pattern, f"test_{i}")
    
    # Test recall
    cue = test_patterns[0].copy()
    mask = np.random.random(cue.shape) > 0.5
    cue[mask] = 0
    
    recalled = substrate.recall_pattern(cue)
    recall_similarity = np.corrcoef(test_patterns[0].flatten(), recalled.flatten())[0, 1]
    
    results['basic_tests']['recall'] = {
        'recall_similarity': float(recall_similarity)
    }
    
    # Test dynamics
    dynamics = substrate.get_dynamics()
    results['basic_tests']['dynamics'] = dynamics
    
    # Save test visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(initial_state, cmap='viridis')
    axes[0, 0].set_title("Initial State")
    
    axes[0, 1].imshow(after_injection, cmap='viridis')
    axes[0, 1].set_title("After Injection")
    
    axes[1, 0].imshow(after_evolution, cmap='viridis')
    axes[1, 0].set_title("After Evolution")
    
    axes[1, 1].imshow(recalled, cmap='viridis')
    axes[1, 1].set_title("Recalled Pattern")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{substrate_name.lower().replace(' ', '_')}_test.png")
    plt.close()
    
    # Save results
    test_file = f"{output_dir}/{substrate_name.lower().replace(' ', '_')}_test.json"
    with open(test_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Test results saved to {test_file}")
    
    return results


def main():
    """Main function to run the experiment."""
    # Set up parameters - using only 3 phoneme patterns
    size = (128, 128)
    num_patterns = 3  # Changed to 3 for phonemes /a/, /i/, /u/
    output_dir = "results/experiment_2a"
    
    # Test individual substrates first
    print("Testing individual substrates...")
    substrates = ["Reaction-Diffusion", "Hopfield Network", "Oscillator Network"]
    
    for substrate_name in substrates:
        try:
            test_individual_substrate(substrate_name, size, output_dir)
            print(f"✓ {substrate_name} test passed")
        except Exception as e:
            print(f"✗ {substrate_name} test failed: {e}")
    
    # Run full comparative experiment
    print("\nRunning full comparative experiment...")
    try:
        results = run_comparative_experiment(size, num_patterns, output_dir)
        print("✓ Comparative experiment completed successfully")
        
        # Print summary
        print("\nExperiment 2A-Rev1 Summary (Phoneme Patterns):")
        for substrate_name, properties in results['results'].items():
            print(f"\n{substrate_name}:")
            print(f"  Capacity: {properties['capacity']:.3f}")
            print(f"  Overlap: {properties['overlap']:.3f}")
            print(f"  Correlation Length: {properties['correlation_length']:.3f}")
            print(f"  Discrimination Accuracy: {properties['discrimination_accuracy']:.3f}")
        
    except Exception as e:
        print(f"✗ Comparative experiment failed: {e}")


if __name__ == "__main__":
    main()