"""
Experiment 1A: Audio Feasibility Test

Tests:
1. Phoneme pattern training (can spectrograms be scarred?)
2. Recall quality (can weak spectrograms be reconstructed?)
3. Discrimination (can substrate distinguish /a/ from /i/ from /u/?)

This tests whether parametric scarring can encode real sensory data.

Success criteria:
- Recall quality > 60%
- Discrimination accuracy > 70%
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import soundfile as sf

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
from core.patterns import create_phoneme_pattern, create_perturbed_pattern

print("=" * 70)
print("EXPERIMENT 1A: AUDIO FEASIBILITY TEST")
print("Can parametric scarring encode phoneme spectrograms?")
print("=" * 70 + "\n")

# Configuration
WIDTH, HEIGHT = 256, 256
SCAR_STRENGTH = 0.003
SIMILARITY_THRESHOLD = 0.5
PHONEMES = ["a", "i", "u"]

# Create output directory
output_dir = Path("results/experiment_1a")
output_dir.mkdir(exist_ok=True, parents=True)
audio_dir = output_dir / "audio_samples"
audio_dir.mkdir(exist_ok=True)

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

print(f"Device: {substrate.device}")
print(f"Grid size: {WIDTH}×{HEIGHT}")
print(f"Phonemes: {PHONEMES}\n")


def train_phoneme_pattern(
    substrate, phoneme, target_pattern, num_iterations=1000, apply_scars=True
):
    """
    Train the substrate to stabilize a phoneme spectrogram pattern.

    Args:
        substrate: GrayScottSubstrate instance
        phoneme: Phoneme identifier ('a', 'i', 'u')
        target_pattern: Spectrogram pattern tensor
        num_iterations: Training steps
        apply_scars: Whether to apply scarring

    Returns:
        tuple: (similarities list, scar_values list)
    """
    print(
        f"Training phoneme /{phoneme}/ {'WITH' if apply_scars else 'WITHOUT'} scarring..."
    )

    # Initialize with weak version of target plus noise
    substrate.V = target_pattern * 0.2 + torch.randn_like(target_pattern) * 0.1
    substrate.V.clamp_(0, 1)
    substrate.U.fill_(1.0)

    similarities = []
    scar_values = []

    for i in range(num_iterations):
        substrate.simulate_step()

        sim = pattern_similarity(substrate.V, target_pattern)
        similarities.append(sim)
        scar_values.append(measure_scar_strength(substrate))

        # Apply scarring when pattern emerges
        if apply_scars:
            apply_scarring_with_quality(
                substrate,
                target_pattern,
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


def test_recall(
    substrate, phoneme, target_pattern, perturbation_strength=0.2, num_steps=500
):
    """
    Test if weak phoneme pattern reconstructs correctly.

    Args:
        substrate: GrayScottSubstrate instance
        phoneme: Phoneme identifier
        target_pattern: Target spectrogram pattern
        perturbation_strength: How weak the initial cue is
        num_steps: Evolution timesteps

    Returns:
        list: Similarity values over time
    """
    print(
        f"Testing recall of /{phoneme}/ with {perturbation_strength * 100:.0f}% perturbation..."
    )

    # Reset to weak perturbation of target
    perturbed = create_perturbed_pattern(target_pattern, perturbation_strength)
    substrate.reset_state(V_init=perturbed)

    similarities = []

    for i in range(num_steps):
        substrate.simulate_step()
        sim = pattern_similarity(substrate.V, target_pattern)
        similarities.append(sim)

        if i % 100 == 0:
            print(f"  Step {i:3d}: Similarity = {sim:.3f}")

    final_sim = similarities[-1]
    print(f"  Final recall quality: {final_sim:.3f}\n")

    return similarities


def test_discrimination(
    substrate, phonemes_dict, perturbation_strength=0.2, num_steps=500
):
    """
    Test if substrate can discriminate between different phonemes.

    For each phoneme:
    - Present weak version
    - Let substrate evolve
    - Measure similarity to ALL phoneme patterns
    - Check if it reconstructs the CORRECT one

    Args:
        substrate: GrayScottSubstrate instance (with scars for all phonemes)
        phonemes_dict: Dict mapping phoneme -> pattern tensor
        perturbation_strength: Initial perturbation
        num_steps: Evolution steps

    Returns:
        dict: Discrimination matrix (which input triggered which pattern)
    """
    print("=" * 70)
    print("DISCRIMINATION TEST")
    print("=" * 70 + "\n")

    discrimination_matrix = {}

    for input_phoneme, input_pattern in phonemes_dict.items():
        print(f"\nTesting input: /{input_phoneme}/")

        # Present weak version
        perturbed = create_perturbed_pattern(input_pattern, perturbation_strength)
        substrate.reset_state(V_init=perturbed)

        # Evolve
        for _ in range(num_steps):
            substrate.simulate_step()

        # Measure similarity to all patterns
        similarities = {}
        for target_phoneme, target_pattern in phonemes_dict.items():
            sim = pattern_similarity(substrate.V, target_pattern)
            similarities[target_phoneme] = sim
            print(f"  Similarity to /{target_phoneme}/: {sim:.3f}")

        # Determine which pattern was reconstructed
        best_match = max(similarities, key=similarities.get)
        discrimination_matrix[input_phoneme] = {
            "similarities": similarities,
            "best_match": best_match,
            "correct": best_match == input_phoneme,
        }

        status = (
            "✓ CORRECT"
            if best_match == input_phoneme
            else f"✗ WRONG (matched /{best_match}/)"
        )
        print(f"  Result: {status}")

    return discrimination_matrix


def run_experiment():
    """Main experimental protocol"""
    results = {}

    # ===== GENERATE PHONEME PATTERNS =====
    print("=" * 70)
    print("STEP 1: GENERATING PHONEME PATTERNS")
    print("=" * 70 + "\n")

    phoneme_data = {}

    for phoneme in PHONEMES:
        print(f"Generating /{phoneme}/...")
        pattern, audio, sample_rate = create_phoneme_pattern(
            phoneme, target_size=(WIDTH, HEIGHT), device=substrate.device, duration=0.5
        )

        # Save audio sample
        audio_path = audio_dir / f"phoneme_{phoneme}.wav"
        sf.write(audio_path, audio, sample_rate)

        # Save pattern visualization
        save_state(pattern, output_dir / f"pattern_{phoneme}.png")

        phoneme_data[phoneme] = {
            "pattern": pattern,
            "audio": audio,
            "sample_rate": sample_rate,
        }

        print(f"  Pattern shape: {pattern.shape}")
        print(f"  Pattern range: [{pattern.min():.3f}, {pattern.max():.3f}]")
        print(f"  Audio saved: {audio_path}\n")

    # ===== PHASE 1: SINGLE PHONEME TRAINING =====
    print("=" * 70)
    print("PHASE 1: SINGLE PHONEME TRAINING (CONTROL)")
    print("Testing if a single phoneme can be scarred")
    print("=" * 70 + "\n")

    test_phoneme = "a"
    target = phoneme_data[test_phoneme]["pattern"]

    # Train WITH scarring
    substrate.reset_parameters()
    train_sim_scarred, train_scars = train_phoneme_pattern(
        substrate, test_phoneme, target, num_iterations=1000, apply_scars=True
    )
    V_trained = substrate.V.clone()
    save_state(V_trained, output_dir / f"after_training_{test_phoneme}_scarred.png")

    # Save scarred parameters
    F_scarred = substrate.F.clone()
    K_scarred = substrate.K.clone()

    # Train WITHOUT scarring (control)
    substrate.reset_parameters()
    train_sim_control, _ = train_phoneme_pattern(
        substrate, test_phoneme, target, num_iterations=1000, apply_scars=False
    )
    V_control = substrate.V.clone()
    save_state(V_control, output_dir / f"after_training_{test_phoneme}_control.png")

    results["phase1"] = {
        "phoneme": test_phoneme,
        "scarred_final_similarity": train_sim_scarred[-1],
        "control_final_similarity": train_sim_control[-1],
        "total_scar_strength": train_scars[-1],
    }

    # ===== PHASE 2: RECALL TEST =====
    print("=" * 70)
    print("PHASE 2: SINGLE PHONEME RECALL")
    print("Testing if weak phoneme reconstructs")
    print("=" * 70 + "\n")

    # Test recall with scarred parameters
    substrate.F.copy_(F_scarred)
    substrate.K.copy_(K_scarred)
    recall_sim_scarred = test_recall(
        substrate, test_phoneme, target, perturbation_strength=0.2
    )
    V_recall_scarred = substrate.V.clone()
    save_state(V_recall_scarred, output_dir / f"recall_{test_phoneme}_scarred.png")

    # Test recall with baseline parameters (control)
    substrate.reset_parameters()
    recall_sim_control = test_recall(
        substrate, test_phoneme, target, perturbation_strength=0.2
    )
    V_recall_control = substrate.V.clone()
    save_state(V_recall_control, output_dir / f"recall_{test_phoneme}_control.png")

    results["phase2"] = {
        "phoneme": test_phoneme,
        "scarred_recall_quality": recall_sim_scarred[-1],
        "control_recall_quality": recall_sim_control[-1],
        "improvement_ratio": recall_sim_scarred[-1] / max(recall_sim_control[-1], 0.01),
    }

    # ===== PHASE 3: MULTI-PHONEME DISCRIMINATION =====
    print("=" * 70)
    print("PHASE 3: MULTI-PHONEME DISCRIMINATION")
    print("Training all 3 phonemes, testing discrimination")
    print("=" * 70 + "\n")

    # Reset substrate and train ALL phonemes
    substrate.reset_parameters()

    phoneme_patterns = {p: phoneme_data[p]["pattern"] for p in PHONEMES}

    for phoneme, pattern in phoneme_patterns.items():
        print(f"\nTraining /{phoneme}/...")
        train_phoneme_pattern(
            substrate, phoneme, pattern, num_iterations=500, apply_scars=True
        )

    # Save parameter scars after all training
    save_parameter_scars(substrate, output_dir / "parameter_scars_all_phonemes.png")

    # Test discrimination
    discrimination_results = test_discrimination(
        substrate, phoneme_patterns, perturbation_strength=0.2, num_steps=500
    )

    # Calculate discrimination accuracy
    correct_count = sum(1 for r in discrimination_results.values() if r["correct"])
    discrimination_accuracy = correct_count / len(discrimination_results)

    results["phase3"] = {
        "discrimination_matrix": {
            k: {
                "similarities": {pk: float(pv) for pk, pv in v["similarities"].items()},
                "best_match": v["best_match"],
                "correct": v["correct"],
            }
            for k, v in discrimination_results.items()
        },
        "discrimination_accuracy": discrimination_accuracy,
        "correct_count": correct_count,
        "total_tests": len(discrimination_results),
    }

    # ===== SAVE RESULTS =====
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")

    # Save results to JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'results.json'}")

    # ===== FINAL REPORT =====
    print("\n" + "=" * 70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print("PHASE 1: Single Phoneme Training")
    print(f"  Phoneme: /{results['phase1']['phoneme']}/")
    print(
        f"  Scarred system final similarity: {results['phase1']['scarred_final_similarity']:.3f}"
    )
    print(
        f"  Control system final similarity: {results['phase1']['control_final_similarity']:.3f}"
    )
    print(f"  Total scar strength: {results['phase1']['total_scar_strength']:.1f}")

    print("\nPHASE 2: Single Phoneme Recall")
    print(
        f"  Scarred recall quality: {results['phase2']['scarred_recall_quality']:.3f}"
    )
    print(
        f"  Control recall quality: {results['phase2']['control_recall_quality']:.3f}"
    )
    print(f"  Improvement ratio: {results['phase2']['improvement_ratio']:.2f}x")

    print("\nPHASE 3: Multi-Phoneme Discrimination")
    print(
        f"  Correct identifications: {results['phase3']['correct_count']}/{results['phase3']['total_tests']}"
    )
    print(
        f"  Discrimination accuracy: {results['phase3']['discrimination_accuracy']:.1%}"
    )

    print("\n  Discrimination Matrix:")
    for input_phoneme, result in results["phase3"]["discrimination_matrix"].items():
        status = "✓" if result["correct"] else "✗"
        print(f"    /{input_phoneme}/ → /{result['best_match']}/ {status}")

    # ===== INTERPRETATION =====
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70 + "\n")

    # Success criteria
    recall_threshold = 0.6
    discrimination_threshold = 0.7

    recall_success = results["phase2"]["scarred_recall_quality"] > recall_threshold
    discrimination_success = (
        results["phase3"]["discrimination_accuracy"] > discrimination_threshold
    )

    if recall_success:
        print(
            f"✓ RECALL SUCCESS: {results['phase2']['scarred_recall_quality']:.1%} > {recall_threshold:.0%}"
        )
    else:
        print(
            f"✗ RECALL FAILURE: {results['phase2']['scarred_recall_quality']:.1%} < {recall_threshold:.0%}"
        )

    if discrimination_success:
        print(
            f"✓ DISCRIMINATION SUCCESS: {results['phase3']['discrimination_accuracy']:.1%} > {discrimination_threshold:.0%}"
        )
    else:
        print(
            f"✗ DISCRIMINATION FAILURE: {results['phase3']['discrimination_accuracy']:.1%} < {discrimination_threshold:.0%}"
        )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70 + "\n")

    if recall_success and discrimination_success:
        print("✓✓ EXPERIMENT SUCCESS")
        print("   Parametric scarring CAN encode phoneme spectrograms.")
        print("   Real sensory data is learnable in this substrate.")
        print("   → Proceed to Experiment 1B (Temporal Sequences)")
    elif recall_success and not discrimination_success:
        print("⚠ PARTIAL SUCCESS")
        print("   Phonemes can be recalled but not discriminated.")
        print("   Possible causes: Patterns too similar, insufficient capacity")
        print("   → Consider: Larger grid, different phonemes, or adjust scarring")
    else:
        print("✗✗ EXPERIMENT FAILURE")
        print("   Parametric scarring cannot reliably encode spectrograms.")
        print("   Sensory data may be too different from geometric patterns.")
        print("   → Pivot to foundational research (Tier 2 experiments)")

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    run_experiment()
