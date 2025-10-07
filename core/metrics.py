"""
metrics.py - Evaluation functions for pattern quality

Key question: How do we measure if two patterns are "similar"?

Different metrics capture different notions of similarity:
- Cosine similarity: Shape/direction (ignores magnitude)
- MSE: Pixel-wise difference (sensitive to intensity)
- SSIM: Structural similarity (perceptual quality)

For now, we use cosine similarity because it's intensity-invariant,
which is important for audio (loud vs quiet phonemes should match).
"""

import torch
import numpy as np


def pattern_similarity(pattern1, pattern2, method="cosine"):
    """
    Measure similarity between two patterns.

    Args:
        pattern1: First pattern (2D tensor)
        pattern2: Second pattern (2D tensor)
        method: Similarity metric ('cosine', 'mse', 'correlation')

    Returns:
        float: Similarity score (higher = more similar)
    """
    if method == "cosine":
        return cosine_similarity(pattern1, pattern2)
    elif method == "mse":
        return mse_similarity(pattern1, pattern2)
    elif method == "correlation":
        return correlation_similarity(pattern1, pattern2)
    else:
        raise ValueError(f"Unknown method: {method}")


def cosine_similarity(pattern1, pattern2):
    """
    Cosine similarity: measures angle between vectors.

    Think of each pattern as a vector in high-dimensional space:
    - Flatten 2D pattern → 1D vector (e.g., 256×256 → 65536 dimensions)
    - Compute angle between the two vectors

    Returns 1 if vectors point in same direction (identical shape)
    Returns 0 if orthogonal (no relationship)
    Returns -1 if opposite directions

    KEY PROPERTY: Ignores magnitude!
    - Pattern with intensity 0.5 and pattern with intensity 1.0
      can have cosine similarity = 1.0 if they have the same shape

    This is why it's good for audio: a loud "ah" and quiet "ah"
    should be recognized as the same phoneme.

    Args:
        pattern1: 2D tensor
        pattern2: 2D tensor

    Returns:
        float: Cosine similarity in [-1, 1]
    """
    # Flatten to 1D vectors
    v1 = pattern1.flatten()
    v2 = pattern2.flatten()

    # Handle edge case: empty or zero patterns
    if v1.sum() < 1e-6 or v2.sum() < 1e-6:
        return 0.0

    # Compute cosine similarity
    # Formula: (v1 · v2) / (||v1|| ||v2||)
    # where · is dot product, || || is magnitude
    similarity = torch.cosine_similarity(v1, v2, dim=0)

    return similarity.item()


def mse_similarity(pattern1, pattern2):
    """
    Mean Squared Error converted to similarity score.

    MSE measures pixel-wise difference:
    - If patterns are identical, MSE = 0
    - If patterns are very different, MSE is large

    We convert to similarity: similarity = 1 / (1 + MSE)
    So similarity ≈ 1 when MSE ≈ 0, and similarity ≈ 0 when MSE is large.

    DIFFERENCE FROM COSINE:
    - MSE cares about absolute intensity
    - A bright pattern vs dim pattern has high MSE even if same shape

    Args:
        pattern1: 2D tensor
        pattern2: 2D tensor

    Returns:
        float: Similarity in [0, 1]
    """
    mse = torch.mean((pattern1 - pattern2) ** 2).item()
    return 1.0 / (1.0 + mse)


def correlation_similarity(pattern1, pattern2):
    """
    Pearson correlation coefficient.

    Like cosine similarity, but also normalizes mean:
    - Centers both patterns around zero (subtracts mean)
    - Then measures linear relationship

    Returns 1 if perfectly correlated
    Returns -1 if perfectly anti-correlated
    Returns 0 if no linear relationship

    Args:
        pattern1: 2D tensor
        pattern2: 2D tensor

    Returns:
        float: Correlation in [-1, 1]
    """
    v1 = pattern1.flatten()
    v2 = pattern2.flatten()

    # Center the vectors
    v1_centered = v1 - v1.mean()
    v2_centered = v2 - v2.mean()

    # Compute correlation
    if v1_centered.std() < 1e-6 or v2_centered.std() < 1e-6:
        return 0.0

    corr = torch.cosine_similarity(v1_centered, v2_centered, dim=0)
    return corr.item()


def recall_quality(substrate, target_pattern, method="cosine"):
    """
    Measure how well current substrate state matches target.

    Convenience function: compares substrate.V with target pattern.

    Args:
        substrate: GrayScottSubstrate instance
        target_pattern: Desired pattern (2D tensor)
        method: Similarity metric

    Returns:
        float: Similarity score
    """
    return pattern_similarity(substrate.V, target_pattern, method=method)


def compute_interference_matrix(substrate, patterns_dict, method="cosine"):
    """
    Measure cross-activation between stored patterns.

    For each pattern A:
    1. Perturb substrate with weak version of A
    2. Let it evolve
    3. Measure similarity to all patterns (A, B, C, ...)

    Result: Matrix showing if pattern A accidentally activates pattern B

    Args:
        substrate: GrayScottSubstrate instance
        patterns_dict: Dict mapping names to patterns
        method: Similarity metric

    Returns:
        numpy array: Interference matrix (N×N)
    """
    pattern_names = list(patterns_dict.keys())
    n_patterns = len(pattern_names)

    interference = np.zeros((n_patterns, n_patterns))

    for i, name_i in enumerate(pattern_names):
        # Perturb with pattern i
        pattern_i = patterns_dict[name_i]
        substrate.reset_state(V_init=pattern_i * 0.15)

        # Evolve for some timesteps
        for _ in range(300):
            substrate.simulate_step()

        # Measure similarity to all patterns
        for j, name_j in enumerate(pattern_names):
            pattern_j = patterns_dict[name_j]
            interference[i, j] = pattern_similarity(
                substrate.V, pattern_j, method=method
            )

    return interference


def discrimination_accuracy(substrate, patterns_dict, num_trials=10):
    """
    Test if substrate can distinguish between patterns.

    For each pattern:
    1. Present weak perturbation
    2. Let substrate evolve
    3. Check if it activates the correct pattern (highest similarity)

    Args:
        substrate: GrayScottSubstrate instance
        patterns_dict: Dict mapping names to patterns
        num_trials: Number of tests per pattern

    Returns:
        float: Accuracy in [0, 1]
    """
    pattern_names = list(patterns_dict.keys())
    correct = 0
    total = 0

    for name in pattern_names:
        target_pattern = patterns_dict[name]

        for _ in range(num_trials):
            # Present weak + noisy version
            substrate.reset_state(
                V_init=target_pattern * 0.2 + torch.randn_like(target_pattern) * 0.05
            )
            substrate.V.clamp_(0, 1)

            # Evolve
            for _ in range(200):
                substrate.simulate_step()

            # Measure similarity to all patterns
            similarities = {
                pname: pattern_similarity(substrate.V, ppattern)
                for pname, ppattern in patterns_dict.items()
            }

            # Check if highest similarity is to correct pattern
            best_match = max(similarities, key=similarities.get)
            if best_match == name:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
