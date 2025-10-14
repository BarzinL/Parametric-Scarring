"""
Experiment 1A-Rev1: Direct Acoustic Excitation

Tests whether RD substrate can extract discriminable features
from raw audio without FFT preprocessing.

Key difference from original 1A:
- Original: FFT → Spectrogram → RD storage
- Revised: Raw audio → Direct injection → Emergent features

Success criteria:
- Pattern distinctness: off-diagonal similarity < 0.6
- Recall accuracy: > 70% correct identifications
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import soundfile as sf
import matplotlib.pyplot as plt

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.substrate import GrayScottSubstrate
from core.scarring import (
    apply_scarring_with_quality,
    measure_scar_strength,
)
from core.metrics import pattern_similarity
from core.visualization import save_state, save_parameter_scars
from core.patterns import synthesize_phoneme_audio, create_circular_region

print("=" * 70)
print("EXPERIMENT 1A-REV1: DIRECT ACOUSTIC EXCITATION")
print("Can RD substrate discover acoustic features without FFT?")
print("=" * 70 + "\n")

# Configuration
WIDTH, HEIGHT = 256, 256
PHONEMES = ["a", "i", "u"]

# Audio parameters
AUDIO_DURATION = 0.5  # seconds
SAMPLE_RATE = 44100   # Hz
INJECTION_STRENGTH = 0.05
AUDIO_STEP_RATIO = 10  # Simulation steps per audio sample

# Injection region
INJECTION_CENTER = (WIDTH // 2, HEIGHT // 2)
INJECTION_RADIUS = 20

# Scarring parameters
SCAR_STRENGTH = 0.003
SIMILARITY_THRESHOLD = 0.5

# Settling parameters
SETTLING_STEPS = 500

# Create output directory
output_dir = Path("results/experiment_1a_rev1")
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
print(f"Injection: center={INJECTION_CENTER}, radius={INJECTION_RADIUS}")
print(f"Audio: {AUDIO_DURATION}s @ {SAMPLE_RATE}Hz")
print(f"Phonemes: {PHONEMES}\n")

# Create injection mask
injection_mask = create_circular_region(
    center=INJECTION_CENTER,
    radius=INJECTION_RADIUS,
    grid_size=(HEIGHT, WIDTH),
    device=substrate.device
)

print(f"Injection region: {injection_mask.sum().item()} pixels\n")


def inject_audio_stream(substrate, audio, injection_mask, strength, step_ratio):
    """
    Stream audio into substrate as temporal perturbations.
    
    Args:
        substrate: GrayScottSubstrate instance
        audio: numpy array of audio samples
        injection_mask: boolean tensor for injection region
        strength: amplitude scaling factor
        step_ratio: simulation steps per audio sample
    
    Returns:
        history: list of V field states at key timesteps
    """
    history = []
    
    for i, sample in enumerate(audio):
        # Inject audio sample into V field
        substrate.V[injection_mask] += float(sample) * strength
        substrate.V.clamp_(0, 1)
        
        # Evolve dynamics
        for _ in range(step_ratio):
            substrate.simulate_step()
        
        # Save snapshots for visualization (every 1000 samples)
        if i % 1000 == 0:
            history.append(substrate.V.clone())
    
    return history


def plot_similarity_matrix(similarity_matrix, phonemes, output_path):
    """
    Create heatmap visualization of phoneme similarity matrix.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy array
    n = len(phonemes)
    matrix = np.zeros((n, n))
    for i, p1 in enumerate(phonemes):
        for j, p2 in enumerate(phonemes):
            matrix[i, j] = similarity_matrix[p1][p2]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'/{p}/' for p in phonemes])
    ax.set_yticklabels([f'/{p}/' for p in phonemes])
    
    # Add values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_title("Phoneme Pattern Similarity Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pattern_comparison(patterns_dict, phonemes, output_path):
    """
    Create side-by-side visualization of emergent patterns.
    """
    fig, axes = plt.subplots(1, len(phonemes), figsize=(15, 5))
    
    for idx, phoneme in enumerate(phonemes):
        pattern = patterns_dict[phoneme].cpu().numpy()
        
        axes[idx].imshow(pattern, cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f'Phoneme /{phoneme}/')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment():
    """Main experimental protocol"""
    results = {}
    
    # ===== PHASE 1: SINGLE PHONEME PATTERN FORMATION =====
    print("=" * 70)
    print("PHASE 1: SINGLE PHONEME PATTERN FORMATION")
    print("Testing if raw audio creates stable spatial patterns")
    print("=" * 70 + "\n")
    
    test_phoneme = "a"
    
    # Synthesize audio
    print(f"Synthesizing /{test_phoneme}/ ({AUDIO_DURATION}s @ {SAMPLE_RATE}Hz)...")
    audio_a, sr = synthesize_phoneme_audio(
        test_phoneme, 
        duration=AUDIO_DURATION, 
        sample_rate=SAMPLE_RATE
    )
    
    # Save audio
    audio_path = audio_dir / f"phoneme_{test_phoneme}.wav"
    sf.write(audio_path, audio_a, sr)
    print(f"  Audio saved: {audio_path}")
    print(f"  Samples: {len(audio_a)}")
    print(f"  Range: [{audio_a.min():.3f}, {audio_a.max():.3f}]\n")
    
    # Reset substrate to equilibrium
    print("Resetting substrate to equilibrium...")
    substrate.reset_to_equilibrium(num_steps=1000)
    print(f"  Initial V std: {torch.std(substrate.V).item():.4f}\n")
    
    # Inject audio
    print("Injecting audio stream...")
    print(f"  Total audio samples: {len(audio_a)}")
    print(f"  Total simulation steps: {len(audio_a) * AUDIO_STEP_RATIO}")
    
    history = inject_audio_stream(
        substrate, 
        audio_a, 
        injection_mask, 
        INJECTION_STRENGTH, 
        AUDIO_STEP_RATIO
    )
    
    print(f"  Injection complete. Captured {len(history)} snapshots.\n")
    
    # Settle
    print(f"Settling for {SETTLING_STEPS} steps...")
    for _ in range(SETTLING_STEPS):
        substrate.simulate_step()
    
    # Analyze emergent pattern
    pattern_a = substrate.V.clone()
    pattern_std = torch.std(pattern_a).item()
    pattern_mean = torch.mean(pattern_a).item()
    
    print(f"  Final V std: {pattern_std:.4f}")
    print(f"  Final V mean: {pattern_mean:.4f}")
    
    # Save pattern
    save_state(pattern_a, output_dir / f"pattern_{test_phoneme}_emergent.png")
    
    results["phase1"] = {
        "phoneme": test_phoneme,
        "pattern_std": pattern_std,
        "pattern_mean": pattern_mean,
        "pattern_emerged": pattern_std > 0.1  # Non-trivial structure
    }
    
    if pattern_std > 0.1:
        print("  ✓ Non-trivial spatial pattern emerged\n")
    else:
        print("  ✗ Pattern too uniform (possible failure)\n")
    
    # ===== PHASE 2: DISCRIMINATION TEST =====
    print("=" * 70)
    print("PHASE 2: MULTI-PHONEME DISCRIMINATION")
    print("Testing if different phonemes create distinct patterns")
    print("=" * 70 + "\n")
    
    emergent_patterns = {}
    
    for phoneme in PHONEMES:
        print(f"Processing /{phoneme}/...")
        
        # Synthesize
        audio, sr = synthesize_phoneme_audio(
            phoneme, 
            duration=AUDIO_DURATION, 
            sample_rate=SAMPLE_RATE
        )
        
        # Save audio
        audio_path = audio_dir / f"phoneme_{phoneme}.wav"
        sf.write(audio_path, audio, sr)
        
        # Reset to equilibrium
        substrate.reset_to_equilibrium(num_steps=1000)
        
        # Inject audio
        inject_audio_stream(
            substrate, 
            audio, 
            injection_mask, 
            INJECTION_STRENGTH, 
            AUDIO_STEP_RATIO
        )
        
        # Settle
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
        # Store pattern
        emergent_patterns[phoneme] = substrate.V.clone()
        save_state(emergent_patterns[phoneme], 
                  output_dir / f"pattern_{phoneme}_emergent.png")
        
        pattern_std = torch.std(emergent_patterns[phoneme]).item()
        print(f"  Pattern std: {pattern_std:.4f}\n")
    
    # Compute similarity matrix
    print("Computing pairwise similarities...")
    similarity_matrix = {}
    
    for p1 in PHONEMES:
        similarity_matrix[p1] = {}
        for p2 in PHONEMES:
            sim = pattern_similarity(emergent_patterns[p1], emergent_patterns[p2])
            similarity_matrix[p1][p2] = float(sim)
            print(f"  similarity(/{p1}/, /{p2}/) = {sim:.3f}")
    
    print()
    
    # Calculate distinctness metric
    diagonal_sum = sum(similarity_matrix[p][p] for p in PHONEMES)
    off_diagonal_sum = sum(
        similarity_matrix[p1][p2] 
        for p1 in PHONEMES 
        for p2 in PHONEMES 
        if p1 != p2
    )
    off_diagonal_avg = off_diagonal_sum / (len(PHONEMES) * (len(PHONEMES) - 1))
    
    print(f"Diagonal average: {diagonal_sum / len(PHONEMES):.3f}")
    print(f"Off-diagonal average: {off_diagonal_avg:.3f}")
    
    patterns_distinct = off_diagonal_avg < 0.6
    
    if patterns_distinct:
        print("✓ Patterns are DISTINCT (off-diagonal < 0.6)\n")
    else:
        print("✗ Patterns are TOO SIMILAR (off-diagonal >= 0.6)\n")
    
    # Save visualizations
    plot_similarity_matrix(similarity_matrix, PHONEMES, 
                          output_dir / "similarity_matrix.png")
    plot_pattern_comparison(emergent_patterns, PHONEMES,
                           output_dir / "emergent_patterns_comparison.png")
    
    results["phase2"] = {
        "similarity_matrix": similarity_matrix,
        "off_diagonal_avg": off_diagonal_avg,
        "patterns_distinct": patterns_distinct
    }
    
    # ===== PHASE 3: SCARRING + RECALL =====
    print("=" * 70)
    print("PHASE 3: SCARRING AND ATTRACTOR-BASED RECALL")
    print("Training substrate on all phonemes, testing discrimination")
    print("=" * 70 + "\n")
    
    # Reset substrate parameters
    substrate.reset_parameters()
    
    # Train on all phonemes sequentially
    for phoneme in PHONEMES:
        print(f"Training on /{phoneme}/...")
        
        audio, sr = synthesize_phoneme_audio(
            phoneme, 
            duration=AUDIO_DURATION, 
            sample_rate=SAMPLE_RATE
        )
        
        # Reset state (keep parameters)
        substrate.reset_state()
        substrate.reset_to_equilibrium(num_steps=500)
        
        # Inject audio with scarring
        for i, sample in enumerate(audio):
            substrate.V[injection_mask] += float(sample) * INJECTION_STRENGTH
            substrate.V.clamp_(0, 1)
            
            for _ in range(AUDIO_STEP_RATIO):
                substrate.simulate_step()
                
                # Apply scarring periodically
                if i % 100 == 0:
                    target = substrate.V.clone()
                    sim = pattern_similarity(substrate.V, target)
                    apply_scarring_with_quality(
                        substrate,
                        target,
                        sim,
                        strength=SCAR_STRENGTH,
                        min_similarity=SIMILARITY_THRESHOLD
                    )
        
        # Settle
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
        total_scars = measure_scar_strength(substrate)
        print(f"  Total scar strength: {total_scars:.1f}\n")
    
    # Save parameter scars
    save_parameter_scars(substrate, output_dir / "parameter_scars_trained.png")
    
    # Test recall: re-present each phoneme
    print("Testing recall...")
    recall_results = {}
    
    for phoneme in PHONEMES:
        print(f"\nRecall test: /{phoneme}/")
        
        audio, sr = synthesize_phoneme_audio(
            phoneme, 
            duration=AUDIO_DURATION, 
            sample_rate=SAMPLE_RATE
        )
        
        # Reset state (keep scars)
        substrate.reset_state()
        substrate.reset_to_equilibrium(num_steps=500)
        
        # Re-inject audio
        inject_audio_stream(
            substrate, 
            audio, 
            injection_mask, 
            INJECTION_STRENGTH, 
            AUDIO_STEP_RATIO
        )
        
        # Settle
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
        # Measure similarity to all reference patterns
        similarities = {}
        for ref_phoneme in PHONEMES:
            sim = pattern_similarity(substrate.V, emergent_patterns[ref_phoneme])
            similarities[ref_phoneme] = float(sim)
            print(f"  Similarity to /{ref_phoneme}/: {sim:.3f}")
        
        best_match = max(similarities, key=similarities.get)
        correct = (best_match == phoneme)
        
        status = "✓ CORRECT" if correct else f"✗ WRONG (matched /{best_match}/)"
        print(f"  Result: {status}")
        
        recall_results[phoneme] = {
            "similarities": similarities,
            "best_match": best_match,
            "correct": correct
        }
    
    # Calculate accuracy
    correct_count = sum(1 for r in recall_results.values() if r["correct"])
    discrimination_accuracy = correct_count / len(recall_results)
    
    print(f"\nDiscrimination accuracy: {discrimination_accuracy:.1%}")
    print(f"Correct: {correct_count}/{len(recall_results)}\n")
    
    results["phase3"] = {
        "recall_results": recall_results,
        "discrimination_accuracy": discrimination_accuracy,
        "correct_count": correct_count,
        "total_tests": len(recall_results)
    }
    
    # ===== SAVE RESULTS =====
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_dir / 'results.json'}\n")
    
    # ===== FINAL REPORT =====
    print("=" * 70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70 + "\n")
    
    print("PHASE 1: Pattern Formation")
    print(f"  Phoneme: /{results['phase1']['phoneme']}/")
    print(f"  Pattern std: {results['phase1']['pattern_std']:.4f}")
    print(f"  Pattern emerged: {results['phase1']['pattern_emerged']}")
    
    print("\nPHASE 2: Discrimination")
    print(f"  Off-diagonal similarity: {results['phase2']['off_diagonal_avg']:.3f}")
    print(f"  Patterns distinct: {results['phase2']['patterns_distinct']}")
    
    print("\nPHASE 3: Recall")
    print(f"  Correct: {results['phase3']['correct_count']}/{results['phase3']['total_tests']}")
    print(f"  Accuracy: {results['phase3']['discrimination_accuracy']:.1%}")
    
    print("\n  Discrimination Matrix:")
    for input_phoneme, result in results['phase3']['recall_results'].items():
        status = "✓" if result['correct'] else "✗"
        print(f"    /{input_phoneme}/ → /{result['best_match']}/ {status}")
    
    # ===== INTERPRETATION =====
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70 + "\n")
    
    pattern_formation_success = results['phase1']['pattern_emerged']
    discrimination_success = results['phase2']['patterns_distinct']
    recall_success = results['phase3']['discrimination_accuracy'] > 0.7
    
    if pattern_formation_success:
        print("✓ PHASE 1 SUCCESS: Spatial patterns emerge from audio")
    else:
        print("✗ PHASE 1 FAILURE: No spatial structure formed")
    
    if discrimination_success:
        print("✓ PHASE 2 SUCCESS: Different phonemes create distinct patterns")
    else:
        print("✗ PHASE 2 FAILURE: Patterns too similar")
    
    if recall_success:
        print("✓ PHASE 3 SUCCESS: Attractor-based discrimination works")
    else:
        print("✗ PHASE 3 FAILURE: Cannot reliably discriminate")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70 + "\n")
    
    if pattern_formation_success and discrimination_success and recall_success:
        print("✓✓ FULL SUCCESS")
        print("   RD substrate CAN extract discriminable acoustic features")
        print("   without FFT preprocessing.")
        print("   → Physics-based feature learning validated")
        print("   → Proceed to temporal sequence learning (Experiment 1B)")
        print("   → Begin PCB analog prototype design")
    elif pattern_formation_success and discrimination_success:
        print("⚠ PARTIAL SUCCESS")
        print("   Features emerge and are distinct, but recall is imperfect.")
        print("   → Refine scarring strategy")
        print("   → Try competitive dynamics (lateral inhibition)")
        print("   → Test with more distinct sounds")
    elif pattern_formation_success:
        print("⚠ MINIMAL SUCCESS")
        print("   Patterns form but lack discriminability.")
        print("   → Substrate responds to audio but cannot extract features")
        print("   → Try: stronger injection, different RD parameters")
        print("   → Consider: preprocessing may be necessary")
    else:
        print("✗✗ EXPERIMENT FAILURE")
        print("   No spatial patterns emerge from raw audio.")
        print("   → Substrate cannot process temporal signals directly")
        print("   → Revert to FFT preprocessing approach (original 1A)")
        print("   → Focus on memory mechanism, not feature extraction")
    
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    run_experiment()