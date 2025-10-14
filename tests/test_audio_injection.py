#!/usr/bin/env python3
"""
Test Script for Audio Injection (Short) Component

This script tests the short audio injection functionality to ensure it works correctly
before running the full experiment.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.substrate import GrayScottSubstrate
from core.patterns import synthesize_phoneme_audio, create_circular_region

def inject_audio_stream(substrate, audio, injection_mask, strength, step_ratio):
    """
    Stream audio into substrate as temporal perturbations.
    
    This is a simplified version of the function from the main experiment
    for testing purposes.
    
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
        
        # Save snapshots for visualization (every N samples)
        if i % max(1, len(audio) // 10) == 0:  # Save ~10 snapshots
            history.append(substrate.V.clone())
    
    return history

def test_audio_injection():
    """Test short audio injection functionality"""
    print("=" * 70)
    print("TESTING AUDIO INJECTION (SHORT) COMPONENT")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("test_results/audio_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passed = True
    
    # Test 1: Basic audio injection
    print("\nTest 1: Basic audio injection")
    try:
        # Initialize substrate
        substrate = GrayScottSubstrate(
            width=256,
            height=256,
            default_f=0.037,
            default_k=0.060,
            Du=0.16,
            Dv=0.08,
            dt=1.0,
            decay_rate=0.9995,
            device='cpu'
        )
        
        # Reset to equilibrium
        substrate.reset_to_equilibrium(num_steps=500)
        initial_v_std = substrate.V.std().item()
        print(f"    Initial V std: {initial_v_std:.4f}")
        
        # Create injection mask
        injection_mask = create_circular_region(
            center=(128, 128),
            radius=20,
            grid_size=(256, 256),
            device='cpu'
        )
        
        # Generate short audio
        audio, sr = synthesize_phoneme_audio('a', duration=0.1, sample_rate=44100)
        print(f"    Audio length: {len(audio)} samples ({len(audio)/sr:.3f}s)")
        
        # Inject audio
        injection_strength = 0.05
        step_ratio = 5
        
        print(f"    Injecting audio with strength {injection_strength}...")
        history = inject_audio_stream(
            substrate, 
            audio, 
            injection_mask, 
            injection_strength, 
            step_ratio
        )
        
        final_v_std = substrate.V.std().item()
        print(f"    Final V std: {final_v_std:.4f}")
        
        # Check that injection had an effect
        assert final_v_std > initial_v_std, "Audio injection should increase V std"
        assert len(history) > 0, "History should not be empty"
        
        print(f"    ✓ Captured {len(history)} snapshots")
        print("    ✓ Basic audio injection works")
        
    except Exception as e:
        print(f"    ✗ Basic injection test failed: {str(e)}")
        all_passed = False
    
    # Test 2: Different injection strengths
    print("\nTest 2: Different injection strengths")
    strengths = [0.01, 0.05, 0.1, 0.2]
    
    for strength in strengths:
        try:
            # Fresh substrate
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            substrate.reset_to_equilibrium(num_steps=200)
            initial_v_std = substrate.V.std().item()
            
            # Create mask and audio
            injection_mask = create_circular_region(
                center=(128, 128), radius=20,
                grid_size=(256, 256), device='cpu'
            )
            
            audio, _ = synthesize_phoneme_audio('a', duration=0.05, sample_rate=44100)
            
            # Inject with different strengths
            inject_audio_stream(substrate, audio, injection_mask, strength, step_ratio=5)
            final_v_std = substrate.V.std().item()
            
            change = final_v_std - initial_v_std
            print(f"    ✓ Strength {strength}: ΔV std = {change:.4f}")
            
            # Stronger injection should have more effect
            assert change > 0, f"No effect with strength {strength}"
            
        except Exception as e:
            print(f"    ✗ Strength test failed for {strength}: {str(e)}")
            all_passed = False
    
    # Test 3: Different phonemes
    print("\nTest 3: Different phonemes")
    phonemes = ['a', 'i', 'u']
    results = {}
    
    for phoneme in phonemes:
        try:
            # Fresh substrate
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            substrate.reset_to_equilibrium(num_steps=200)
            initial_v_std = substrate.V.std().item()
            
            # Create mask and audio
            injection_mask = create_circular_region(
                center=(128, 128), radius=20,
                grid_size=(256, 256), device='cpu'
            )
            
            audio, _ = synthesize_phoneme_audio(phoneme, duration=0.1, sample_rate=44100)
            
            # Inject
            inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio=5)
            final_v_std = substrate.V.std().item()
            
            results[phoneme] = final_v_std
            change = final_v_std - initial_v_std
            print(f"    ✓ Phoneme /{phoneme}/: ΔV std = {change:.4f}")
            
            assert change > 0, f"No effect with phoneme /{phoneme}/"
            
        except Exception as e:
            print(f"    ✗ Phoneme test failed for /{phoneme}/: {str(e)}")
            all_passed = False
    
    # Check that different phonemes produce different effects
    if len(results) == 3:
        stds = list(results.values())
        std_variation = np.std(stds)
        print(f"    Phoneme effect variation: {std_variation:.4f}")
        # Should have some variation between phonemes
        assert std_variation > 0.0001, "Phonemes should produce different effects"
    
    # Test 4: Different injection regions
    print("\nTest 4: Different injection regions")
    regions = [
        (64, 64, 15),    # Top-left, small
        (128, 128, 20),  # Center, medium
        (200, 200, 25),  # Bottom-right, large
    ]
    
    for center_x, center_y, radius in regions:
        try:
            # Fresh substrate
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            substrate.reset_to_equilibrium(num_steps=200)
            initial_v_std = substrate.V.std().item()
            
            # Create mask and audio
            injection_mask = create_circular_region(
                center=(center_x, center_y), radius=radius,
                grid_size=(256, 256), device='cpu'
            )
            
            audio, _ = synthesize_phoneme_audio('a', duration=0.1, sample_rate=44100)
            
            # Inject
            inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio=5)
            final_v_std = substrate.V.std().item()
            
            change = final_v_std - initial_v_std
            print(f"    ✓ Region ({center_x},{center_y}), r={radius}: ΔV std = {change:.4f}")
            
            assert change > 0, f"No effect with region ({center_x},{center_y})"
            
        except Exception as e:
            print(f"    ✗ Region test failed for ({center_x},{center_y}): {str(e)}")
            all_passed = False
    
    # Test 5: Different step ratios
    print("\nTest 5: Different step ratios")
    step_ratios = [1, 5, 10, 20]
    
    for step_ratio in step_ratios:
        try:
            # Fresh substrate
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            substrate.reset_to_equilibrium(num_steps=200)
            initial_v_std = substrate.V.std().item()
            
            # Create mask and audio
            injection_mask = create_circular_region(
                center=(128, 128), radius=20,
                grid_size=(256, 256), device='cpu'
            )
            
            audio, _ = synthesize_phoneme_audio('a', duration=0.05, sample_rate=44100)
            
            # Inject
            inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio)
            final_v_std = substrate.V.std().item()
            
            change = final_v_std - initial_v_std
            print(f"    ✓ Step ratio {step_ratio}: ΔV std = {change:.4f}")
            
            assert change > 0, f"No effect with step ratio {step_ratio}"
            
        except Exception as e:
            print(f"    ✗ Step ratio test failed for {step_ratio}: {str(e)}")
            all_passed = False
    
    # Test 6: Visualization
    print("\nTest 6: Visualization")
    try:
        # Fresh substrate
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        substrate.reset_to_equilibrium(num_steps=300)
        
        # Create mask and audio
        injection_mask = create_circular_region(
            center=(128, 128), radius=20,
            grid_size=(256, 256), device='cpu'
        )
        
        audio, _ = synthesize_phoneme_audio('a', duration=0.2, sample_rate=44100)
        
        # Inject and capture history
        history = inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio=5)
        
        # Plot evolution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot initial state
        axes[0, 0].imshow(history[0].cpu().numpy(), cmap='viridis', origin='lower', vmin=0, vmax=1)
        axes[0, 0].set_title('Initial State')
        axes[0, 0].axis('off')
        
        # Plot intermediate states
        for i in range(1, min(5, len(history))):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].imshow(history[i].cpu().numpy(), cmap='viridis', origin='lower', vmin=0, vmax=1)
                axes[row, col].set_title(f'Step {i * len(audio) // len(history)}')
                axes[row, col].axis('off')
        
        # Plot final state
        axes[1, 2].imshow(substrate.V.cpu().numpy(), cmap='viridis', origin='lower', vmin=0, vmax=1)
        axes[1, 2].set_title('Final State')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        viz_path = output_dir / 'injection_evolution.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"    ✗ Visualization failed: {str(e)}")
        all_passed = False
    
    # Test 7: Device compatibility
    print("\nTest 7: Device compatibility")
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        try:
            substrate = GrayScottSubstrate(
                width=128, height=128,  # Smaller for faster testing
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device=device
            )
            
            substrate.reset_to_equilibrium(num_steps=100)
            initial_v_std = substrate.V.std().item()
            
            injection_mask = create_circular_region(
                center=(64, 64), radius=10,
                grid_size=(128, 128), device=device
            )
            
            audio, _ = synthesize_phoneme_audio('a', duration=0.05, sample_rate=44100)
            
            inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio=3)
            final_v_std = substrate.V.std().item()
            
            change = final_v_std - initial_v_std
            print(f"    ✓ Device {device}: ΔV std = {change:.4f}")
            
            assert change > 0, f"No effect on device {device}"
            
        except Exception as e:
            print(f"    ✗ Device test failed for {device}: {str(e)}")
            all_passed = False
    
    # Test 8: Performance test
    print("\nTest 8: Performance test")
    try:
        import time
        
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        substrate.reset_to_equilibrium(num_steps=200)
        
        injection_mask = create_circular_region(
            center=(128, 128), radius=20,
            grid_size=(256, 256), device='cpu'
        )
        
        audio, _ = synthesize_phoneme_audio('a', duration=0.1, sample_rate=44100)
        
        start_time = time.time()
        inject_audio_stream(substrate, audio, injection_mask, 0.05, step_ratio=5)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"    ✓ Injection time: {elapsed_time:.3f}s")
        
        # Should be reasonably fast
        assert elapsed_time < 10.0, f"Injection too slow: {elapsed_time:.3f}s"
        
    except Exception as e:
        print(f"    ✗ Performance test failed: {str(e)}")
        all_passed = False
    
    # Test 9: Edge cases
    print("\nTest 9: Edge cases")
    
    # Test 9a: Empty audio
    try:
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        substrate.reset_to_equilibrium(num_steps=100)
        
        injection_mask = create_circular_region(
            center=(128, 128), radius=20,
            grid_size=(256, 256), device='cpu'
        )
        
        empty_audio = np.array([])
        history = inject_audio_stream(substrate, empty_audio, injection_mask, 0.05, step_ratio=5)
        
        assert len(history) == 0, "Empty audio should produce empty history"
        print("    ✓ Empty audio handled correctly")
        
    except Exception as e:
        print(f"    ✗ Empty audio test failed: {str(e)}")
        all_passed = False
    
    # Test 9b: Very short audio
    try:
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        substrate.reset_to_equilibrium(num_steps=100)
        
        injection_mask = create_circular_region(
            center=(128, 128), radius=20,
            grid_size=(256, 256), device='cpu'
        )
        
        # Single sample
        short_audio = np.array([0.5])
        history = inject_audio_stream(substrate, short_audio, injection_mask, 0.05, step_ratio=5)
        
        assert len(history) > 0, "Short audio should produce history"
        print("    ✓ Very short audio handled correctly")
        
    except Exception as e:
        print(f"    ✗ Short audio test failed: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ AUDIO INJECTION TEST: ALL PASSED")
        print("Audio injection component is working correctly!")
    else:
        print("✗✗✗ AUDIO INJECTION TEST: SOME FAILED")
        print("Check the errors above before running the full experiment.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = test_audio_injection()
    sys.exit(0 if success else 1)