"""
scarring.py - Memory formation through parameter modification

"Scarring" means: when a pattern successfully forms, we modify the substrate's
physical parameters (F and K) to make that pattern easier to form in the future.

This is fundamentally different from weight-based learning:
- Neural networks: Store patterns in connection weights
- Parametric scarring: Store patterns in the physics itself
"""

import torch


def apply_scarring(substrate, target_pattern, strength=0.003, threshold=0.3):
    """
    Modify substrate parameters where current state matches target.

    The scarring rule:
    1. Find regions where both V and target are active (match_map)
    2. Increase F (feed rate) in those regions
    3. Decrease K (kill rate) in those regions
    4. Keep F and K within valid bounds

    This makes the pattern "easier" to form - the physics now favors it.

    Args:
        substrate: GrayScottSubstrate instance
        target_pattern: Desired pattern (2D tensor, same shape as substrate.V)
        strength: How much to modify F/K (larger = stronger scars)
        threshold: Activation threshold for matching (0-1)
    """
    # Match map: where both current state V and target are "active"
    # This identifies regions where the pattern is currently present
    match_map = (substrate.V > threshold) & (target_pattern > threshold)

    # Modify F: Increase feed rate where pattern exists
    # Higher F means more U is replenished, stabilizing V patterns
    substrate.F[match_map] += strength

    # Modify K: Decrease kill rate where pattern exists
    # Lower K means V persists longer, reinforcing the pattern
    substrate.K[match_map] -= strength * 0.5

    # Clamp to valid parameter ranges
    # (Outside these bounds, Gray-Scott dynamics become unstable)
    substrate.F.clamp_(0.01, 0.08)
    substrate.K.clamp_(0.03, 0.08)


def apply_scarring_with_quality(
    substrate,
    target_pattern,
    current_similarity,
    strength=0.003,
    threshold=0.3,
    min_similarity=0.5,
):
    """
    Apply scarring only if pattern quality exceeds threshold.

    This prevents "weak" or incorrect patterns from being scarred.
    Only reinforce patterns when they're actually correct.

    Args:
        substrate: GrayScottSubstrate instance
        target_pattern: Desired pattern
        current_similarity: How well current state matches target (0-1)
        strength: Scarring strength
        threshold: Activation threshold
        min_similarity: Only scar if similarity > this value

    Returns:
        bool: True if scarring was applied
    """
    if current_similarity > min_similarity:
        apply_scarring(substrate, target_pattern, strength, threshold)
        return True
    return False


def apply_temporal_scarring(substrate, pattern_A, pattern_B, strength=0.002):
    """
    Scar transition pathways between two patterns (A → B).

    EXPERIMENTAL: Not yet validated.

    Idea: Modify parameters in regions that are:
    - Active in pattern A (source)
    - Will become active in pattern B (target)

    This should create a "pathway" where A automatically triggers B.

    Args:
        substrate: GrayScottSubstrate instance
        pattern_A: Source pattern
        pattern_B: Target pattern
        strength: Scarring strength
    """
    # Transition zone: Where A is high and B will be high
    # This is speculative - may need different logic
    transition_map = (pattern_A > 0.3) & (pattern_B > 0.3)

    # Modify parameters to favor B when A is present
    substrate.F[transition_map] += strength
    substrate.K[transition_map] -= strength * 0.3

    substrate.F.clamp_(0.01, 0.08)
    substrate.K.clamp_(0.03, 0.08)


def measure_scar_strength(substrate):
    """
    Quantify total scarring (divergence from baseline).

    Args:
        substrate: GrayScottSubstrate instance

    Returns:
        float: Sum of absolute deviations in F and K
    """
    return substrate.get_scar_divergence()


def visualize_scars(substrate):
    """
    Get F and K deviations for visualization.

    Args:
        substrate: GrayScottSubstrate instance

    Returns:
        tuple: (F_change, K_change) as numpy arrays
    """
    f_change = (substrate.F - substrate.default_f).cpu().numpy()
    k_change = (substrate.K - substrate.default_k).cpu().numpy()
    return f_change, k_change


def clear_scars(substrate):
    """
    Erase all scars, resetting to default physics.

    Args:
        substrate: GrayScottSubstrate instance
    """
    substrate.reset_parameters()
