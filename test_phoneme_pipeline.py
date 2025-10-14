"""
Quick test of phoneme generation pipeline
Tests that audio synthesis → FFT → pattern conversion works correctly
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf

sys.path.append(str(Path(__file__).parent))

from core.patterns import (
    create_phoneme_pattern,
    synthesize_vowel_phoneme,
    audio_to_spectrogram,
)

print("=== Testing Phoneme Generation Pipeline ===\n")

# Test parameters
phonemes = ["a", "i", "u"]
output_dir = Path("test_audio_output")
output_dir.mkdir(exist_ok=True)

# Generate and visualize each phoneme
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, phoneme in enumerate(phonemes):
    print(f"Generating phoneme /{phoneme}/...")

    # Generate pattern
    pattern, audio, sample_rate = create_phoneme_pattern(
        phoneme, target_size=(256, 256), device="cpu", duration=0.5
    )

    # Save audio file
    audio_path = output_dir / f"phoneme_{phoneme}.wav"
    sf.write(audio_path, audio, sample_rate)
    print(f"  Saved audio: {audio_path}")

    # Get spectrogram for visualization
    spectrogram = audio_to_spectrogram(audio, sample_rate)

    # Plot waveform
    axes[0, idx].plot(audio[:1000])  # First 1000 samples
    axes[0, idx].set_title(f"/{phoneme}/ Waveform")
    axes[0, idx].set_xlabel("Sample")
    axes[0, idx].set_ylabel("Amplitude")

    # Plot spectrogram (raw)
    im = axes[1, idx].imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1, idx].set_title(f"/{phoneme}/ Spectrogram (raw)")
    axes[1, idx].set_xlabel("Time")
    axes[1, idx].set_ylabel("Frequency")
    plt.colorbar(im, ax=axes[1, idx], label="Magnitude (dB)")

    print(f"  Pattern shape: {pattern.shape}")
    print(f"  Pattern range: [{pattern.min():.3f}, {pattern.max():.3f}]")
    print(f"  Audio length: {len(audio)} samples ({len(audio) / sample_rate:.3f}s)")
    print(f"  Spectrogram shape: {spectrogram.shape}\n")

plt.tight_layout()
output_fig = output_dir / "phoneme_comparison.png"
plt.savefig(output_fig, dpi=150, bbox_inches="tight")
print(f"Saved visualization: {output_fig}")

# Now plot the substrate-ready patterns
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

for idx, phoneme in enumerate(phonemes):
    pattern, _, _ = create_phoneme_pattern(
        phoneme, target_size=(256, 256), device="cpu"
    )

    axes2[idx].imshow(pattern.cpu().numpy(), cmap="viridis", origin="lower")
    axes2[idx].set_title(f"/{phoneme}/ Pattern (256×256)")
    axes2[idx].set_xlabel("X")
    axes2[idx].set_ylabel("Y")

plt.tight_layout()
output_fig2 = output_dir / "patterns_substrate_ready.png"
plt.savefig(output_fig2, dpi=150, bbox_inches="tight")
print(f"Saved substrate patterns: {output_fig2}")

print("\n=== Pipeline Test Complete ===")
print(f"Check {output_dir}/ for generated files:")
print(f"  - 3 WAV files (playable audio)")
print(f"  - 2 PNG visualizations")
