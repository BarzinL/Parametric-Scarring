#!/usr/bin/env python3
"""
Test Script for Audio Synthesis Component

This script tests the phoneme audio synthesis functionality to ensure it works correctly
before running the full experiment.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.patterns import synthesize_phoneme_audio

def test_audio_synthesis():
    """Test audio synthesis for all phonemes"""
    print("=" * 70)
    print("TESTING AUDIO SYNTHESIS COMPONENT")
    print("=" * 70)
    
    # Test parameters
    phonemes = ["a", "i", "u"]
    duration = 0.5  # seconds
    sample_rate = 44100  # Hz
    
    # Create output directory
    output_dir = Path("test_results/audio_synthesis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passed = True
    
    for phoneme in phonemes:
        print(f"\nTesting phoneme /{phoneme}/...")
        
        try:
            # Test 1: Basic synthesis
            print("  Test 1: Basic synthesis")
            audio, sr = synthesize_phoneme_audio(
                phoneme, 
                duration=duration, 
                sample_rate=sample_rate
            )
            
            # Validate basic properties
            assert sr == sample_rate, f"Sample rate mismatch: {sr} != {sample_rate}"
            assert len(audio) == int(sample_rate * duration), f"Audio length mismatch: {len(audio)} != {int(sample_rate * duration)}"
            assert np.max(np.abs(audio)) <= 1.0, f"Audio not normalized: max={np.max(np.abs(audio))}"
            assert np.min(audio) >= -1.0, f"Audio values below -1: min={np.min(audio)}"
            
            print(f"    ✓ Length: {len(audio)} samples")
            print(f"    ✓ Duration: {len(audio)/sr:.3f}s")
            print(f"    ✓ Range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
            
            # Test 2: Save audio file
            print("  Test 2: Saving audio file")
            audio_path = output_dir / f"phoneme_{phoneme}.wav"
            sf.write(audio_path, audio, sr)
            assert audio_path.exists(), "Audio file not saved"
            print(f"    ✓ Saved to: {audio_path}")
            
            # Test 3: Different durations
            print("  Test 3: Different durations")
            for test_duration in [0.2, 0.5, 1.0]:
                test_audio, test_sr = synthesize_phoneme_audio(
                    phoneme, 
                    duration=test_duration, 
                    sample_rate=sample_rate
                )
                expected_length = int(sample_rate * test_duration)
                assert len(test_audio) == expected_length, f"Duration test failed for {test_duration}s"
                print(f"    ✓ {test_duration}s: {len(test_audio)} samples")
            
            # Test 4: Different sample rates
            print("  Test 4: Different sample rates")
            for test_sr in [22050, 44100, 48000]:
                test_audio, test_sr_returned = synthesize_phoneme_audio(
                    phoneme, 
                    duration=0.5, 
                    sample_rate=test_sr
                )
                assert test_sr_returned == test_sr, f"Sample rate mismatch: {test_sr_returned} != {test_sr}"
                assert len(test_audio) == int(test_sr * 0.5), f"Audio length mismatch for SR {test_sr}"
                print(f"    ✓ SR {test_sr}: {len(test_audio)} samples")
            
            # Test 5: Formant characteristics (basic check)
            print("  Test 5: Formant characteristics")
            # Check that different phonemes produce different signals
            if phoneme == "a":  # Store first phoneme for comparison
                reference_audio = audio.copy()
            else:
                # Calculate correlation with reference
                correlation = np.corrcoef(reference_audio, audio)[0, 1]
                print(f"    ✓ Correlation with /a/: {correlation:.3f}")
                # Should be different but not completely uncorrelated
                assert 0.3 < correlation < 0.95, f"Unusual correlation: {correlation}"
            
            print(f"  ✓ All tests passed for /{phoneme}/")
            
        except Exception as e:
            print(f"  ✗ Test failed for /{phoneme}/: {str(e)}")
            all_passed = False
    
    # Test 6: Error handling
    print("\nTest 6: Error handling")
    try:
        bad_audio, _ = synthesize_phoneme_audio("x", duration=0.5, sample_rate=44100)
        print("  ✗ Should have raised error for invalid phoneme")
        all_passed = False
    except ValueError:
        print("  ✓ Correctly raised ValueError for invalid phoneme")
    except Exception as e:
        print(f"  ✗ Unexpected error: {str(e)}")
        all_passed = False
    
    # Test 7: Visualization
    print("\nTest 7: Creating visualization")
    try:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        
        for idx, phoneme in enumerate(phonemes):
            audio, sr = synthesize_phoneme_audio(phoneme, duration=0.5, sample_rate=44100)
            
            # Waveform
            time_axis = np.linspace(0, len(audio)/sr, len(audio))
            axes[idx, 0].plot(time_axis, audio)
            axes[idx, 0].set_title(f'Phoneme /{phoneme}/ Waveform')
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 0].set_ylabel('Amplitude')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # FFT spectrum
            fft = np.fft.fft(audio)
            freq_axis = np.fft.fftfreq(len(audio), 1/sr)
            positive_freqs = freq_axis[:len(freq_axis)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            axes[idx, 1].plot(positive_freqs, positive_fft)
            axes[idx, 1].set_title(f'Phoneme /{phoneme}/ Spectrum')
            axes[idx, 1].set_xlabel('Frequency (Hz)')
            axes[idx, 1].set_ylabel('Magnitude')
            axes[idx, 1].set_xlim(0, 5000)  # Focus on low frequencies
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = output_dir / "phoneme_comparison.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"  ✗ Visualization failed: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ AUDIO SYNTHESIS TEST: ALL PASSED")
        print("Audio synthesis component is working correctly!")
    else:
        print("✗✗✗ AUDIO SYNTHESIS TEST: SOME FAILED")
        print("Check the errors above before running the full experiment.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = test_audio_synthesis()
    sys.exit(0 if success else 1)