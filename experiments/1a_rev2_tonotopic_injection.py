"""
Experiment 1A-Rev2: Tonotopic (Frequency-Mapped) Injection

Key innovation: Different frequency bands inject at different spatial locations,
mimicking cochlear frequency-to-position mapping.

Hypothesis: Spatial separation of frequency components will create
discriminable patterns even without explicit scarring selectivity.
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sys.path.append(str(Path(__file__).parent.parent))

from core.substrate import GrayScottSubstrate
from core.scarring import apply_scarring_with_quality, measure_scar_strength
from core.metrics import pattern_similarity
from core.visualization import save_state, save_parameter_scars
from core.patterns import synthesize_phoneme_audio, create_circular_region

print("=" * 70)
print("EXPERIMENT 1A-REV2: TONOTOPIC INJECTION")
print("Spatially-distributed frequency encoding")
print("=" * 70 + "\n")

# Configuration
WIDTH, HEIGHT = 256, 256
PHONEMES = ["a", "i", "u"]

# Audio parameters
AUDIO_DURATION = 0.5
SAMPLE_RATE = 44100
INJECTION_STRENGTH = 0.1  # Increased from 0.05
AUDIO_STEP_RATIO = 10

# Frequency bands and spatial mapping
FREQUENCY_BANDS = [
    (0, 500, 32),        # Very low → far left
    (500, 1000, 64),     # Low → left
    (1000, 2000, 128),   # Mid → center
    (2000, 4000, 192),   # High → right
    (4000, 8000, 224),   # Very high → far right
]

INJECTION_RADIUS = 15
Y_CENTER = HEIGHT // 2

# Scarring parameters
SCAR_STRENGTH = 0.003
SIMILARITY_THRESHOLD = 0.5
SETTLING_STEPS = 500

# Output
output_dir = Path("results/experiment_1a_rev2")
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
print(f"Tonotopic mapping: {len(FREQUENCY_BANDS)} frequency bands")
print(f"Phonemes: {PHONEMES}\n")


def decompose_into_frequency_bands(audio, sample_rate, bands):
    """
    Decompose audio into frequency bands using bandpass filters.
    
    Args:
        audio: Input waveform (numpy array)
        sample_rate: Sampling rate in Hz
        bands: List of (low_freq, high_freq, x_position) tuples
    
    Returns:
        dict: {x_position: filtered_audio_segment}
    """
    nyquist = sample_rate / 2
    filtered_signals = {}
    
    for low_freq, high_freq, x_pos in bands:
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Prevent edge cases
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(0.01, min(high_norm, 0.99))
        
        # Design bandpass filter
        if low_freq == 0:
            # Low-pass filter
            b, a = butter(4, high_norm, btype='low')
        elif high_freq >= nyquist:
            # High-pass filter
            b, a = butter(4, low_norm, btype='high')
        else:
            # Bandpass filter
            b, a = butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter
        filtered = filtfilt(b, a, audio)
        filtered_signals[x_pos] = filtered
    
    return filtered_signals


def create_tonotopic_injection_masks(bands, width, height, radius, device):
    """
    Create injection masks for each frequency band position.
    
    Returns:
        dict: {x_position: boolean_mask}
    """
    masks = {}
    
    for _, _, x_pos in bands:
        mask = create_circular_region(
            center=(x_pos, height // 2),
            radius=radius,
            grid_size=(height, width),
            device=device
        )
        masks[x_pos] = mask
    
    return masks


def inject_tonotopic_audio(substrate, audio, sample_rate, bands, masks, 
                           strength, step_ratio):
    """
    Inject audio with frequency-to-space mapping.
    
    Different frequency bands are injected at different spatial locations,
    mimicking cochlear tonotopic organization.
    """
    # Decompose audio into frequency bands
    filtered_signals = decompose_into_frequency_bands(audio, sample_rate, bands)
    
    # Find longest signal for iteration
    max_length = max(len(sig) for sig in filtered_signals.values())
    
    for i in range(max_length):
        # Inject each frequency band at its designated position
        for x_pos, filtered_audio in filtered_signals.items():
            if i < len(filtered_audio):
                sample = filtered_audio[i]
                substrate.V[masks[x_pos]] += float(sample) * strength
        
        # Clamp to valid range
        substrate.V.clamp_(0, 1)
        
        # Evolve dynamics
        for _ in range(step_ratio):
            substrate.simulate_step()


def visualize_tonotopic_map(bands, width, height, output_path):
    """
    Visualize which frequency bands map to which spatial positions.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Draw substrate grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Draw injection regions
    for low_freq, high_freq, x_pos in bands:
        circle = plt.Circle((x_pos, height // 2), INJECTION_RADIUS, 
                           color='red', alpha=0.3)
        ax.add_patch(circle)
        
        # Label
        label = f"{low_freq}-{high_freq}Hz"
        ax.text(x_pos, height // 2 - 40, label, 
               ha='center', fontsize=8)
    
    # Format
    ax.set_xlabel("Spatial Position (pixels)")
    ax.set_ylabel("Grid Height (pixels)")
    ax.set_title("Tonotopic Frequency-to-Space Mapping")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_matrix(similarity_matrix, phonemes, output_path):
    """Create heatmap visualization of phoneme similarity matrix."""
    n = len(phonemes)
    matrix = np.zeros((n, n))
    for i, p1 in enumerate(phonemes):
        for j, p2 in enumerate(phonemes):
            matrix[i, j] = similarity_matrix[p1][p2]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'/{p}/' for p in phonemes])
    ax.set_yticklabels([f'/{p}/' for p in phonemes])
    
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_title("Phoneme Pattern Similarity Matrix (Tonotopic)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pattern_comparison(patterns_dict, phonemes, output_path):
    """Create side-by-side visualization of emergent patterns."""
    fig, axes = plt.subplots(1, len(phonemes), figsize=(15, 5))
    
    for idx, phoneme in enumerate(phonemes):
        pattern = patterns_dict[phoneme].cpu().numpy()
        
        axes[idx].imshow(pattern, cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f'Phoneme /{phoneme}/ (Tonotopic)', fontsize=12)
        axes[idx].axis('off')
        
        # Add frequency info
        if phoneme == 'a':
            freq_text = "F1=700Hz, F2=1200Hz"
        elif phoneme == 'i':
            freq_text = "F1=300Hz, F2=2300Hz"
        else:  # u
            freq_text = "F1=300Hz, F2=900Hz"
        
        axes[idx].text(128, 240, freq_text, ha='center', 
                      fontsize=8, color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment():
    """Main experimental protocol"""
    results = {}
    
    # Create injection masks
    print("Creating tonotopic injection masks...")
    injection_masks = create_tonotopic_injection_masks(
        FREQUENCY_BANDS, WIDTH, HEIGHT, INJECTION_RADIUS, substrate.device
    )
    
    total_injection_pixels = sum(mask.sum().item() for mask in injection_masks.values())
    print(f"  Total injection pixels: {total_injection_pixels}")
    print(f"  Coverage: {total_injection_pixels / (WIDTH * HEIGHT) * 100:.1f}%\n")
    
    # Visualize tonotopic map
    visualize_tonotopic_map(FREQUENCY_BANDS, WIDTH, HEIGHT,
                           output_dir / "tonotopic_mapping.png")
    
    # ===== PHASE 1: SINGLE PHONEME PATTERN FORMATION =====
    print("=" * 70)
    print("PHASE 1: SINGLE PHONEME WITH TONOTOPIC INJECTION")
    print("=" * 70 + "\n")
    
    test_phoneme = "a"
    
    print(f"Synthesizing /{test_phoneme}/...")
    audio_a, sr = synthesize_phoneme_audio(test_phoneme, AUDIO_DURATION, SAMPLE_RATE)
    audio_path = audio_dir / f"phoneme_{test_phoneme}.wav"
    sf.write(audio_path, audio_a, sr)
    print(f"  Audio saved: {audio_path}\n")
    
    print("Resetting substrate...")
    substrate.reset_to_equilibrium(num_steps=1000)
    print(f"  Initial V std: {torch.std(substrate.V).item():.4f}\n")
    
    print("Injecting with tonotopic mapping...")
    inject_tonotopic_audio(
        substrate, audio_a, SAMPLE_RATE, 
        FREQUENCY_BANDS, injection_masks,
        INJECTION_STRENGTH, AUDIO_STEP_RATIO
    )
    
    print(f"Settling for {SETTLING_STEPS} steps...")
    for _ in range(SETTLING_STEPS):
        substrate.simulate_step()
    
    pattern_a = substrate.V.clone()
    pattern_std = torch.std(pattern_a).item()
    pattern_mean = torch.mean(pattern_a).item()
    
    print(f"  Final V std: {pattern_std:.4f}")
    print(f"  Final V mean: {pattern_mean:.4f}")
    
    save_state(pattern_a, output_dir / f"pattern_{test_phoneme}_tonotopic.png")
    
    results["phase1"] = {
        "phoneme": test_phoneme,
        "pattern_std": pattern_std,
        "pattern_mean": pattern_mean,
        "pattern_emerged": pattern_std > 0.1
    }
    
    if pattern_std > 0.1:
        print("  ✓ Non-trivial spatial pattern emerged\n")
    else:
        print("  ✗ Pattern too uniform\n")
    
    # ===== PHASE 2: DISCRIMINATION TEST =====
    print("=" * 70)
    print("PHASE 2: MULTI-PHONEME DISCRIMINATION (TONOTOPIC)")
    print("=" * 70 + "\n")
    
    emergent_patterns = {}
    
    for phoneme in PHONEMES:
        print(f"Processing /{phoneme}/...")
        
        audio, sr = synthesize_phoneme_audio(phoneme, AUDIO_DURATION, SAMPLE_RATE)
        audio_path = audio_dir / f"phoneme_{phoneme}.wav"
        sf.write(audio_path, audio, sr)
        
        substrate.reset_to_equilibrium(num_steps=1000)
        
        inject_tonotopic_audio(
            substrate, audio, SAMPLE_RATE,
            FREQUENCY_BANDS, injection_masks,
            INJECTION_STRENGTH, AUDIO_STEP_RATIO
        )
        
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
        emergent_patterns[phoneme] = substrate.V.clone()
        save_state(emergent_patterns[phoneme],
                  output_dir / f"pattern_{phoneme}_tonotopic.png")
        
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
    
    # Calculate distinctness
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
        print("✗ Patterns still too similar (off-diagonal >= 0.6)\n")
    
    plot_similarity_matrix(similarity_matrix, PHONEMES,
                          output_dir / "similarity_matrix_tonotopic.png")
    plot_pattern_comparison(emergent_patterns, PHONEMES,
                           output_dir / "emergent_patterns_tonotopic.png")
    
    results["phase2"] = {
        "similarity_matrix": similarity_matrix,
        "off_diagonal_avg": off_diagonal_avg,
        "patterns_distinct": patterns_distinct
    }
    
    # ===== PHASE 3: SCARRING + RECALL =====
    print("=" * 70)
    print("PHASE 3: SCARRING AND RECALL (TONOTOPIC)")
    print("=" * 70 + "\n")
    
    substrate.reset_parameters()
    
    for phoneme in PHONEMES:
        print(f"Training on /{phoneme}/...")
        
        audio, sr = synthesize_phoneme_audio(phoneme, AUDIO_DURATION, SAMPLE_RATE)
        
        substrate.reset_state()
        substrate.reset_to_equilibrium(num_steps=500)
        
        # Inject with periodic scarring
        filtered_signals = decompose_into_frequency_bands(audio, SAMPLE_RATE, FREQUENCY_BANDS)
        max_length = max(len(sig) for sig in filtered_signals.values())
        
        for i in range(max_length):
            for x_pos, filtered_audio in filtered_signals.items():
                if i < len(filtered_audio):
                    substrate.V[injection_masks[x_pos]] += float(filtered_audio[i]) * INJECTION_STRENGTH
            
            substrate.V.clamp_(0, 1)
            
            for _ in range(AUDIO_STEP_RATIO):
                substrate.simulate_step()
            
            # Apply scarring periodically
            if i % 100 == 0:
                target = substrate.V.clone()
                sim = pattern_similarity(substrate.V, target)
                apply_scarring_with_quality(
                    substrate, target, sim,
                    strength=SCAR_STRENGTH,
                    min_similarity=SIMILARITY_THRESHOLD
                )
        
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
        total_scars = measure_scar_strength(substrate)
        print(f"  Total scar strength: {total_scars:.1f}\n")
    
    save_parameter_scars(substrate, output_dir / "parameter_scars_tonotopic.png")
    
    # Test recall
    print("Testing recall...")
    recall_results = {}
    
    for phoneme in PHONEMES:
        print(f"\nRecall test: /{phoneme}/")
        
        audio, sr = synthesize_phoneme_audio(phoneme, AUDIO_DURATION, SAMPLE_RATE)
        
        substrate.reset_state()
        substrate.reset_to_equilibrium(num_steps=500)
        
        inject_tonotopic_audio(
            substrate, audio, SAMPLE_RATE,
            FREQUENCY_BANDS, injection_masks,
            INJECTION_STRENGTH, AUDIO_STEP_RATIO
        )
        
        for _ in range(SETTLING_STEPS):
            substrate.simulate_step()
        
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
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70 + "\n")
    
    print("PHASE 1: Pattern Formation")
    print(f"  Pattern emerged: {results['phase1']['pattern_emerged']}")
    
    print("\nPHASE 2: Discrimination")
    print(f"  Off-diagonal similarity: {results['phase2']['off_diagonal_avg']:.3f}")
    print(f"  Patterns distinct: {results['phase2']['patterns_distinct']}")
    
    print("\nPHASE 3: Recall")
    print(f"  Accuracy: {results['phase3']['discrimination_accuracy']:.1%}")
    print(f"  Correct: {results['phase3']['correct_count']}/{results['phase3']['total_tests']}")
    
    print("\n  Discrimination Matrix:")
    for phoneme, result in results['phase3']['recall_results'].items():
        status = "✓" if result['correct'] else "✗"
        print(f"    /{phoneme}/ → /{result['best_match']}/ {status}")
    
    # ===== INTERPRETATION =====
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70 + "\n")
    
    if results['phase2']['patterns_distinct'] and results['phase3']['discrimination_accuracy'] > 0.7:
        print("✓✓ FULL SUCCESS")
        print("   Tonotopic encoding enables discrimination!")
        print("   → Spatial frequency mapping is critical")
        print("   → Physics-based feature extraction validated")
    elif results['phase2']['patterns_distinct']:
        print("⚠ PARTIAL SUCCESS")
        print("   Patterns distinct but recall imperfect")
        print("   → Tonotopic encoding works")
        print("   → Scarring needs refinement")
    else:
        print("✗ TONOTOPIC ENCODING INSUFFICIENT")
        print("   Patterns still too similar")
        print("   → Need stronger injection or different parameters")
    
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    run_experiment()