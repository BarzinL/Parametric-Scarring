"""
Property measurement functions for Experiment 2A.

This module provides functions to measure and compare properties
across different computational substrates.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def measure_capacity(substrate, patterns: List[np.ndarray]) -> float:
    """
    Measure the storage capacity of a substrate.
    
    Args:
        substrate: Computational substrate instance
        patterns: List of patterns to test storage capacity
        
    Returns:
        Storage capacity metric (0.0 to 1.0)
    """
    # Reset substrate
    substrate.reset()
    
    # Store patterns one by one
    stored_patterns = []
    for i, pattern in enumerate(patterns):
        substrate.store_pattern(pattern, f"pattern_{i}")
        stored_patterns.append(pattern)
    
    # Test recall for each pattern
    recall_scores = []
    for i, pattern in enumerate(stored_patterns):
        # Create partial cue (e.g., mask 50% of pattern)
        cue = pattern.copy()
        mask = np.random.random(pattern.shape) > 0.5
        cue[mask] = 0
        
        # Recall pattern
        recalled = substrate.recall_pattern(cue)
        
        # Calculate similarity
        similarity = np.corrcoef(pattern.flatten(), recalled.flatten())[0, 1]
        recall_scores.append(similarity)
    
    # Capacity is average recall success rate
    return np.mean(recall_scores)


def measure_overlap(substrate, patterns: List[np.ndarray]) -> float:
    """
    Measure pattern overlap in a substrate.
    
    Args:
        substrate: Computational substrate instance
        patterns: List of patterns to test
        
    Returns:
        Pattern overlap metric (0.0 to 1.0)
    """
    # Reset substrate
    substrate.reset()
    
    # Store all patterns
    for i, pattern in enumerate(patterns):
        substrate.store_pattern(pattern, f"pattern_{i}")
    
    # Measure overlap by injecting each pattern and checking response
    overlap_scores = []
    
    for i, pattern in enumerate(patterns):
        # Inject pattern and let substrate settle
        substrate.inject_pattern(pattern)
        substrate.evolve(steps=20)
        
        # Get final state
        final_state = substrate.get_state()
        
        # Calculate overlap with original pattern
        overlap = np.corrcoef(pattern.flatten(), final_state.flatten())[0, 1]
        overlap_scores.append(overlap)
    
    return np.mean(overlap_scores)


def measure_correlation_length(substrate, state: Optional[np.ndarray] = None) -> float:
    """
    Measure the correlation length of substrate state.
    
    Args:
        substrate: Computational substrate instance
        state: Optional state to analyze (uses current state if None)
        
    Returns:
        Correlation length metric
    """
    if state is None:
        state = substrate.get_state()
    
    # Calculate spatial autocorrelation
    # Use 2D FFT to compute correlation function
    fft_state = np.fft.fft2(state)
    autocorr = np.fft.ifft2(fft_state * np.conj(fft_state)).real
    
    # Normalize
    autocorr = autocorr / autocorr[0, 0]
    
    # Find correlation length (distance where autocorrelation drops to 1/e)
    center = np.array(autocorr.shape) // 2
    
    # Create radial profile
    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Bin by radius
    max_radius = min(center)
    radial_profile = np.zeros(max_radius)
    
    for radius in range(max_radius):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = autocorr[mask].mean()
    
    # Find correlation length
    threshold = 1.0 / np.e
    correlation_length = 0
    
    for i, value in enumerate(radial_profile):
        if value < threshold:
            correlation_length = i
            break
    
    return correlation_length


def measure_dynamics(substrate, steps: int = 100) -> Dict[str, float]:
    """
    Measure dynamical properties of a substrate.
    
    Args:
        substrate: Computational substrate instance
        steps: Number of evolution steps to analyze
        
    Returns:
        Dictionary of dynamical metrics
    """
    # Reset substrate
    substrate.reset()
    
    # Add small random perturbation
    noise = np.random.random(substrate.size) * 0.1
    substrate.inject_pattern(noise)
    
    # Track evolution
    states = []
    dynamics_history = []
    
    for _ in range(steps):
        substrate.evolve(steps=1)
        states.append(substrate.get_state().copy())
        dynamics_history.append(substrate.get_dynamics())
    
    # Calculate dynamical metrics
    states = np.array(states)
    
    # Measure convergence rate
    state_changes = []
    for i in range(1, len(states)):
        change = np.mean((states[i] - states[i-1])**2)
        state_changes.append(change)
    
    convergence_rate = np.mean(state_changes[-10:]) if len(state_changes) > 10 else 0
    
    # Measure stability (variance in final states)
    stability = 1.0 / (1.0 + np.var(states[-10:]) if len(states) > 10 else 1.0)
    
    # Measure oscillation frequency (if any)
    if len(states) > 20:
        # Simple frequency estimation using FFT
        mean_activity = [np.mean(state) for state in states]
        fft_activity = np.fft.fft(mean_activity)
        dominant_freq = np.argmax(np.abs(fft_activity[1:len(fft_activity)//2])) + 1
    else:
        dominant_freq = 0
    
    return {
        'convergence_rate': convergence_rate,
        'stability': stability,
        'dominant_frequency': dominant_freq,
        'final_state_variance': np.var(states[-10:]) if len(states) > 10 else 0
    }


def compute_similarity_matrix(substrate, patterns: List[np.ndarray]) -> np.ndarray:
    """
    Compute similarity matrix for patterns in a substrate.
    
    Args:
        substrate: Computational substrate instance
        patterns: List of patterns to compare
        
    Returns:
        Similarity matrix (N x N)
    """
    n_patterns = len(patterns)
    similarity_matrix = np.zeros((n_patterns, n_patterns))
    
    # Reset substrate
    substrate.reset()
    
    # Store all patterns
    for i, pattern in enumerate(patterns):
        substrate.store_pattern(pattern, f"pattern_{i}")
    
    # Compute pairwise similarities
    for i in range(n_patterns):
        for j in range(n_patterns):
            # Use pattern i as cue
            cue = patterns[i].copy()
            
            # Recall and compare with pattern j
            recalled = substrate.recall_pattern(cue)
            similarity = np.corrcoef(patterns[j].flatten(), recalled.flatten())[0, 1]
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix


def analyze_discrimination_performance(substrate, patterns: List[np.ndarray], 
                                     labels: List[str]) -> Dict[str, Any]:
    """
    Analyze discrimination performance of a substrate.
    
    Args:
        substrate: Computational substrate instance
        patterns: List of patterns to discriminate
        labels: List of pattern labels
        
    Returns:
        Dictionary of discrimination metrics
    """
    n_patterns = len(patterns)
    confusion_matrix = np.zeros((n_patterns, n_patterns))
    
    # Reset substrate
    substrate.reset()
    
    # Store all patterns
    for i, pattern in enumerate(patterns):
        substrate.store_pattern(pattern, labels[i])
    
    # Test discrimination
    for i, pattern in enumerate(patterns):
        # Create partial cue
        cue = pattern.copy()
        mask = np.random.random(pattern.shape) > 0.3  # Keep 70% of pattern
        cue[mask] = 0
        
        # Recall pattern
        recalled = substrate.recall_pattern(cue)
        
        # Compare with all stored patterns
        similarities = []
        for j, stored_pattern in enumerate(patterns):
            similarity = np.corrcoef(stored_pattern.flatten(), recalled.flatten())[0, 1]
            similarities.append(similarity)
        
        # Find best match
        best_match = np.argmax(similarities)
        confusion_matrix[i, best_match] += 1
    
    # Calculate metrics
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    # Calculate precision and recall for each class
    precisions = []
    recalls = []
    
    for i in range(n_patterns):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return {
        'accuracy': accuracy,
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'confusion_matrix': confusion_matrix,
        'per_class_precision': precisions,
        'per_class_recall': recalls
    }


def measure_substrate_properties(substrate, test_patterns: List[np.ndarray]) -> Dict[str, float]:
    """
    Measure comprehensive properties of a substrate.
    
    Args:
        substrate: Computational substrate instance
        test_patterns: List of patterns to test with
        
    Returns:
        Dictionary of measured properties
    """
    properties = {}
    
    # Measure capacity
    properties['capacity'] = measure_capacity(substrate, test_patterns)
    
    # Measure overlap
    properties['overlap'] = measure_overlap(substrate, test_patterns)
    
    # Measure correlation length
    properties['correlation_length'] = measure_correlation_length(substrate)
    
    # Measure dynamics
    dynamics = measure_dynamics(substrate)
    properties.update(dynamics)
    
    # Measure discrimination performance
    labels = [f"pattern_{i}" for i in range(len(test_patterns))]
    discrimination = analyze_discrimination_performance(substrate, test_patterns, labels)
    properties['discrimination_accuracy'] = discrimination['accuracy']
    
    return properties


def compare_substrates(substrates: Dict[str, Any], test_patterns: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compare properties across multiple substrates.
    
    Args:
        substrates: Dictionary of substrate instances
        test_patterns: List of patterns to test with
        
    Returns:
        Dictionary of properties for each substrate
    """
    results = {}
    
    for name, substrate in substrates.items():
        print(f"Measuring properties for {name}...")
        properties = measure_substrate_properties(substrate, test_patterns)
        results[name] = properties
    
    return results