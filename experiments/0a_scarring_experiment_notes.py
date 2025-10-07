# Goal:
# Can F/K modifications create stable, reactivatable memory patterns?
#
# Hypothesis: If scarring works as memory, then:
# 1. Training a pattern should modify local F/K values.
# 2. After training, that pattern should reactivate more easily than untrained patterns.
# 3. Multiple patterns should be storable without complete interference.
#
# Test Protocol:
# Phase 1: Single Pattern Training (Proof of Concept)
# - Define a target pattern (e.g. 4 bright spots in a square arrangement)
# - Apply scarring when the manifold approximates this pattern
# - Test: After 1000 frames, does a weak seed near that pattern grow toward the target?
# - Control: Does the same seed prodcuce different results in an unscarred region?
#
# Phase 2: Pattern Recall
# - Let the system evolve for 500 frames after training
# - Introduce a weak perturbation (10% of original seed)
# - Measure: Does the full pattern reconstruct? (memory recall)
#
# Phase 3: Capacity Test
# - Train 3 different patterns in different spatial regions
# - Test recall for each independently
# - Measure interference (does recalling pattern A corrupt pattern B?)
#
# Phase 4: Generalization
# - Train on pattern A
# - Test with slightly modified version (rotated, scaled, translated)
# - Does the scar generalize or is it position-specific?

import torch


# Metrics
# We need quantitative measures:
def pattern_similarity(V, target_pattern, region):
    """Measure how closely V matches target in a region"""
    return torch.cosine_similarity(V[region].flatten(), target_pattern.flatten(), dim=0)


def scar_strength(F, K, baseline_F, baseline_K):
    """Total divergence from baseline parameters"""
    return torch.abs(F - baseline_F).sum() + torch.abs(K - baseline_K).sum()


def recall_quality(initial_perturbation, final_state, target):
    """How well did weak seed reconstruct target?"""
    return pattern_similarity(final_state, target) / initial_perturbation.mean()


# Implementation Strategy
# Start minimal. Here's what we need:


# 1. Target Pattern Generator
def create_target_pattern(pattern_type="square_spots"):
    target = torch.zeros(HEIGHT, WIDTH, device=device)
    if pattern_type == "square_spots":
        positions = [(128, 128), (128, 384), (384, 128), (384, 384)]
        for x, y in positions:
            mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 < 20**2
            target[mask] = 0.8
    return target


# 2. Scarring Function
def apply_scarring(V, target, F, K, strength=0.01):
    """Scar F/K to stabilize current pattern toward target"""
    # Where V is high and matches target, increase F slightly
    match_map = (V > 0.3) & (target > 0.3)
    F[match_map] += strength
    K[match_map] -= strength * 0.5

    # Clamp to valid ranges
    F.clamp_(0.01, 0.08)
    K.clamp_(0.03, 0.08)


# 3. Training Loop
def train_pattern(target, num_iterations=1000):
    # Initialize with weak version of target
    V[:] = target * 0.2 + torch.randn_like(target) * 0.1
    V.clamp_(0, 1)


    for i in range(num_iterations):
        # Simulate reaction-diffusion
        simulate_step()

        # Measure similarity to target
        similarity = pattern_similarity(V, target, slice(None))

        # Apply scarring when pattern emerges
        if similarity > 0.5:
            apply_scarring(V, target, F, K)

        if i % 100 == 0:
            print(f"Iteration {i}: Similarity = {similarity:.3f}")


# 4. Recall Test
def test_recall(target, perturbation_strength=0.1):
    # Reset V to weak perturbation
    V[:] = target * perturbation_strength
    V += torch.randn_like(V) * 0.05
    V.clamp_(0, 1)

    similarities = []
    for i in range(500):
        simulate_step()
        sim = pattern_similarity(V, target, slice(None))
        similarities.append(sim)

    # Did it converge to target?
    final_similarity = similarities[-1]
    return final_similarity, similarities


# Success Criteria
# Minimum viable result:
# - After training, recall quality > 0.7 (70% match to target)
# - Untrained patterns show recall quality < 0.3
# - At least 2 different patterns can coexist with < 20% interference
#
# Strong result:
# - Recall quality > 0.85
# - 3+ patterns storable
# - Patterns show some generalization (rotated versions partially recall)
#
#
# What This Tells Us
# If it works:
# - Scarring is a viable memory mechanism
# - We can proceed to more complex experiments (sequential patterns, multimodal binding)
# - The substrate has representational capacity
#
# If it partially works:
# - We learn the limits (capacity, interference characteristics)
# - We can refine the scarring algorithm
# We identify what additional mechanisms are needed
#
# If it fails:
# - We've saved years of pursuing a dead end
# - We can pivot to alternative substrates (NCA, reservoir computing, etc.)
# - We've learned about reaction-diffusion limitations
#
# Timeline:
# - Week 1: Implement training/testing infrastructure, run Phase 1
# - Week 2: Analyze results, tune parameters, run Phases 2-3
# - Week 3: Document findings, create visualizations
# - Week 4: Write up results (blog post, paper draft, or detailed notes)
