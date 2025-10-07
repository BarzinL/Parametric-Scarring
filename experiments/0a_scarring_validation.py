"""
Scarring Validation Experiment (Phases 1-3)

Tests:
1. Single pattern training (with/without scarring)
2. Memory recall from weak perturbation
3. Multi-pattern capacity

This is the REFACTORED version using modular core components.
"""

import torch
import json
from pathlib import Path
import sys

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.substrate import GrayScottSubstrate
from core.scarring import (
    apply_scarring_with_quality,
    measure_scar_strength,
    visualize_scars,
)
from core.metrics import recall_quality, pattern_similarity
from core.visualization import save_state, save_parameter_scars, plot_experiment_summary
from core.patterns import create_geometric_pattern, create_perturbed_pattern

print("=== Scarring Validation Experiment (Refactored) ===\n")

# Configuration
WIDTH, HEIGHT = 256, 256
SCAR_STRENGTH = 0.003
SIMILARITY_THRESHOLD = 0.5

# Create output directory
output_dir = Path("scarring_results")
output_dir.mkdir(exist_ok=True)

# Initialize substrate
substrate = GrayScottSubstrate(
    width=WIDTH,
    height=HEIGHT,
    default_f=0.037,
    default_k=0.060,
    Du=0.16,
    Dv=0.08,
    dt=1.0,
    decay_rate=0.9995,
)

print(f"Using device: {substrate.device}\n")


def train_pattern(substrate, target, num_iterations=1000, apply_scars=True):
    """
    Train the system to produce a target pattern.

    Args:
        substrate: GrayScottSubstrate instance
        target: Target pattern
        num_iterations: Training steps
        apply_scars: Whether to apply scarring

    Returns:
        tuple: (similarities list, scar_values list)
    """
    print(f"Training {'WITH' if apply_scars else 'WITHOUT'} scarring...")

    # Initialize with weak version of target plus noise
    substrate.V = target * 0.2 + torch.randn_like(target) * 0.1
    substrate.V.clamp_(0, 1)
    substrate.U.fill_(1.0)

    similarities = []
    scar_values = []

    for i in range(num_iterations):
        substrate.simulate_step()

        sim = pattern_similarity(substrate.V, target)
        similarities.append(sim)
        scar_values.append(measure_scar_strength(substrate))

        # Apply scarring when pattern emerges well
        if apply_scars:
            apply_scarring_with_quality(
                substrate,
                target,
                sim,
                strength=SCAR_STRENGTH,
                min_similarity=SIMILARITY_THRESHOLD,
            )

        if i % 200 == 0:
            print(
                f"  Iteration {i:4d}: Similarity = {sim:.3f}, Scars = {scar_values[-1]:.1f}"
            )

    final_sim = similarities[-1]
    print(f"  Final similarity: {final_sim:.3f}\n")

    return similarities, scar_values


def test_recall(substrate, target, perturbation_strength=0.15, num_steps=500):
    """
    Test if weak perturbation reconstructs the pattern.

    Args:
        substrate: GrayScottSubstrate instance
        target: Target pattern
        perturbation_strength: How weak the initial cue is
        num_steps: Evolution timesteps

    Returns:
        list: Similarity values over time
    """
    print(f"Testing recall with {perturbation_strength * 100:.0f}% perturbation...")

    # Reset to weak perturbation of target
    perturbed = create_perturbed_pattern(target, perturbation_strength)
    substrate.reset_state(V_init=perturbed)

    similarities = []

    for i in range(num_steps):
        substrate.simulate_step()
        sim = pattern_similarity(substrate.V, target)
        similarities.append(sim)

        if i % 100 == 0:
            print(f"  Step {i:3d}: Similarity = {sim:.3f}")

    final_sim = similarities[-1]
    print(f"  Final recall quality: {final_sim:.3f}\n")

    return similarities


def run_experiment():
    """Main experimental protocol"""
    results = {}

    # ===== PHASE 1: Single Pattern Training =====
    print("=" * 50)
    print("PHASE 1: Single Pattern Training")
    print("=" * 50 + "\n")

    target = create_geometric_pattern("square_spots", WIDTH, HEIGHT, substrate.device)
    save_state(target, output_dir / "target_pattern.png")

    # Train WITH scarring
    substrate.reset_parameters()
    train_sim_scarred, train_scars = train_pattern(
        substrate, target, num_iterations=1000, apply_scars=True
    )
    V_trained = substrate.V.clone()
    save_state(V_trained, output_dir / "after_training_scarred.png")

    # Save scarred parameters
    F_scarred = substrate.F.clone()
    K_scarred = substrate.K.clone()

    # Train WITHOUT scarring (control)
    substrate.reset_parameters()
    train_sim_control, _ = train_pattern(
        substrate, target, num_iterations=1000, apply_scars=False
    )
    V_control = substrate.V.clone()
    save_state(V_control, output_dir / "after_training_control.png")

    results["phase1"] = {
        "scarred_final_similarity": train_sim_scarred[-1],
        "control_final_similarity": train_sim_control[-1],
        "total_scar_strength": train_scars[-1],
    }

    # ===== PHASE 2: Pattern Recall =====
    print("=" * 50)
    print("PHASE 2: Pattern Recall Test")
    print("=" * 50 + "\n")

    # Test recall with scarred parameters
    substrate.F.copy_(F_scarred)
    substrate.K.copy_(K_scarred)
    recall_sim_scarred = test_recall(substrate, target, perturbation_strength=0.15)
    V_recall_scarred = substrate.V.clone()
    save_state(V_recall_scarred, output_dir / "recall_scarred.png")

    # Test recall with baseline parameters (control)
    substrate.reset_parameters()
    recall_sim_control = test_recall(substrate, target, perturbation_strength=0.15)
    V_recall_control = substrate.V.clone()
    save_state(V_recall_control, output_dir / "recall_control.png")

    results["phase2"] = {
        "scarred_recall_quality": recall_sim_scarred[-1],
        "control_recall_quality": recall_sim_control[-1],
        "improvement_ratio": recall_sim_scarred[-1] / max(recall_sim_control[-1], 0.01),
    }

    # ===== PHASE 3: Multiple Patterns =====
    print("=" * 50)
    print("PHASE 3: Multiple Pattern Capacity")
    print("=" * 50 + "\n")

    substrate.reset_parameters()

    patterns = {
        "pattern_A": create_geometric_pattern(
            "square_spots", WIDTH, HEIGHT, substrate.device, region_offset=(-64, -64)
        ),
        "pattern_B": create_geometric_pattern(
            "ring", WIDTH, HEIGHT, substrate.device, region_offset=(0, 0)
        ),
        "pattern_C": create_geometric_pattern(
            "line", WIDTH, HEIGHT, substrate.device, region_offset=(0, 0)
        ),
    }

    # Train all three patterns sequentially
    for name, pattern in patterns.items():
        print(f"\nTraining {name}...")
        train_pattern(substrate, pattern, num_iterations=500, apply_scars=True)

    # Save parameter scars after all training
    save_parameter_scars(substrate, output_dir / "parameter_scars.png")

    # Test recall for each
    recall_qualities = {}
    for name, pattern in patterns.items():
        print(f"\nRecalling {name}...")
        recall_sim = test_recall(
            substrate, pattern, perturbation_strength=0.2, num_steps=300
        )
        recall_qualities[name] = recall_sim[-1]
        save_state(substrate.V, output_dir / f"recall_{name}.png")

    results["phase3"] = {
        "pattern_recall_qualities": recall_qualities,
        "average_recall": sum(recall_qualities.values()) / len(recall_qualities),
        "min_recall": min(recall_qualities.values()),
    }

    # ===== Generate Summary Plots =====
    print("\n" + "=" * 50)
    print("Generating summary plots...")
    print("=" * 50 + "\n")

    plot_experiment_summary(
        {
            "training_scarred": train_sim_scarred,
            "training_control": train_sim_control,
            "scar_accumulation": train_scars,
            "recall_scarred": recall_sim_scarred,
            "recall_control": recall_sim_control,
            "multi_pattern_qualities": recall_qualities,
        },
        output_dir,
    )

    # Save results to JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ===== Final Report =====
    print("\n" + "=" * 50)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 50 + "\n")

    print("PHASE 1: Single Pattern Training")
    print(
        f"  Scarred system final similarity: {results['phase1']['scarred_final_similarity']:.3f}"
    )
    print(
        f"  Control system final similarity: {results['phase1']['control_final_similarity']:.3f}"
    )
    print(f"  Total scar strength: {results['phase1']['total_scar_strength']:.1f}")

    print("\nPHASE 2: Pattern Recall")
    print(
        f"  Scarred recall quality: {results['phase2']['scarred_recall_quality']:.3f}"
    )
    print(
        f"  Control recall quality: {results['phase2']['control_recall_quality']:.3f}"
    )
    print(f"  Improvement ratio: {results['phase2']['improvement_ratio']:.2f}x")

    print("\nPHASE 3: Multiple Pattern Capacity")
    for name, quality in results["phase3"]["pattern_recall_qualities"].items():
        print(f"  {name}: {quality:.3f}")
    print(f"  Average recall: {results['phase3']['average_recall']:.3f}")
    print(f"  Minimum recall: {results['phase3']['min_recall']:.3f}")

    print("\n" + "=" * 50)
    print("INTERPRETATION")
    print("=" * 50 + "\n")

    # Success criteria evaluation
    success_threshold = 0.7
    improvement_threshold = 1.5

    if results["phase2"]["scarred_recall_quality"] > success_threshold:
        print("✓ SUCCESS: Scarred system shows strong recall (>0.7)")
    else:
        print("✗ PARTIAL: Scarred recall below success threshold")

    if results["phase2"]["improvement_ratio"] > improvement_threshold:
        print(
            f"✓ SUCCESS: {results['phase2']['improvement_ratio']:.1f}x improvement over baseline"
        )
    else:
        print("✗ PARTIAL: Scarring doesn't provide substantial advantage")

    if results["phase3"]["min_recall"] > 0.5:
        print("✓ SUCCESS: Multiple patterns can coexist")
    else:
        print("✗ FAILURE: Pattern interference too high")

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    run_experiment()
