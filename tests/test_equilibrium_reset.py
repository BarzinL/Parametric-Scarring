#!/usr/bin/env python3
"""
Test Script for Equilibrium Reset Component

This script tests the substrate equilibrium reset functionality to ensure it works correctly
before running the full experiment.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.substrate import GrayScottSubstrate

def test_equilibrium_reset():
    """Test substrate equilibrium reset functionality"""
    print("=" * 70)
    print("TESTING EQUILIBRIUM RESET COMPONENT")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("test_results/equilibrium_reset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passed = True
    
    # Test 1: Basic reset functionality
    print("\nTest 1: Basic reset functionality")
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
        
        # Check initial state
        initial_u_mean = substrate.U.mean().item()
        initial_v_mean = substrate.V.mean().item()
        initial_u_std = substrate.U.std().item()
        initial_v_std = substrate.V.std().item()
        
        print(f"    Initial U: mean={initial_u_mean:.4f}, std={initial_u_std:.4f}")
        print(f"    Initial V: mean={initial_v_mean:.4f}, std={initial_v_std:.4f}")
        
        # Verify initial conditions
        assert abs(initial_u_mean - 1.0) < 0.01, f"Initial U should be ~1.0: {initial_u_mean}"
        assert abs(initial_v_mean - 0.0) < 0.01, f"Initial V should be ~0.0: {initial_v_mean}"
        assert initial_u_std < 0.01, f"Initial U should be uniform: {initial_u_std}"
        assert initial_v_std < 0.01, f"Initial V should be uniform: {initial_v_std}"
        
        # Perform reset to equilibrium
        print("    Performing reset_to_equilibrium...")
        substrate.reset_to_equilibrium(num_steps=100)
        
        # Check state after reset
        reset_u_mean = substrate.U.mean().item()
        reset_v_mean = substrate.V.mean().item()
        reset_u_std = substrate.U.std().item()
        reset_v_std = substrate.V.std().item()
        
        print(f"    Reset U: mean={reset_u_mean:.4f}, std={reset_u_std:.4f}")
        print(f"    Reset V: mean={reset_v_mean:.4f}, std={reset_v_std:.4f}")
        
        # V should have some structure after reset (not uniform)
        assert reset_v_std > 0.01, f"V should have structure after reset: {reset_v_std}"
        assert reset_u_mean > 0.9, f"U should remain high after reset: {reset_u_mean}"
        
        print("    ✓ Basic reset functionality works")
        
    except Exception as e:
        print(f"    ✗ Basic reset test failed: {str(e)}")
        all_passed = False
    
    # Test 2: Different reset durations
    print("\nTest 2: Different reset durations")
    durations = [50, 100, 500, 1000]
    
    for duration in durations:
        try:
            # Create fresh substrate
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=0.037, default_k=0.060,
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            # Reset with different durations
            substrate.reset_to_equilibrium(num_steps=duration)
            
            v_std = substrate.V.std().item()
            u_mean = substrate.U.mean().item()
            
            print(f"    ✓ Duration {duration}: V std={v_std:.4f}, U mean={u_mean:.4f}")
            
            # Should have some structure for all durations
            assert v_std > 0.01, f"No structure after {duration} steps"
            assert 0.8 < u_mean < 1.0, f"U mean out of range: {u_mean}"
            
        except Exception as e:
            print(f"    ✗ Duration test failed for {duration}: {str(e)}")
            all_passed = False
    
    # Test 3: Reset after perturbation
    print("\nTest 3: Reset after perturbation")
    try:
        # Create substrate
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        # Add strong perturbation
        substrate.V[100:150, 100:150] = 0.8
        substrate.V[50:80, 50:80] = 0.6
        substrate.V[180:220, 180:220] = 0.7
        
        perturbed_v_std = substrate.V.std().item()
        print(f"    Perturbed V std: {perturbed_v_std:.4f}")
        
        # Reset
        substrate.reset_to_equilibrium(num_steps=500)
        
        reset_v_std = substrate.V.std().item()
        reset_u_mean = substrate.U.mean().item()
        
        print(f"    After reset V std: {reset_v_std:.4f}")
        print(f"    After reset U mean: {reset_u_mean:.4f}")
        
        # Should return to structured state
        assert reset_v_std > 0.01, "No structure after reset from perturbation"
        assert 0.8 < reset_u_mean < 1.0, "U mean out of range after reset"
        
        print("    ✓ Reset after perturbation works")
        
    except Exception as e:
        print(f"    ✗ Reset after perturbation failed: {str(e)}")
        all_passed = False
    
    # Test 4: Different substrate parameters
    print("\nTest 4: Different substrate parameters")
    parameter_sets = [
        {"f": 0.037, "k": 0.060},  # Default
        {"f": 0.026, "k": 0.051},  # Spots
        {"f": 0.039, "k": 0.058},  # Stripes
    ]
    
    for params in parameter_sets:
        try:
            substrate = GrayScottSubstrate(
                width=256, height=256,
                default_f=params["f"], default_k=params["k"],
                Du=0.16, Dv=0.08, dt=1.0,
                decay_rate=0.9995, device='cpu'
            )
            
            substrate.reset_to_equilibrium(num_steps=500)
            
            v_std = substrate.V.std().item()
            u_mean = substrate.U.mean().item()
            
            print(f"    ✓ f={params['f']}, k={params['k']}: V std={v_std:.4f}, U mean={u_mean:.4f}")
            
            assert v_std > 0.01, f"No structure for f={params['f']}, k={params['k']}"
            assert 0.7 < u_mean < 1.0, f"U mean out of range for f={params['f']}, k={params['k']}"
            
        except Exception as e:
            print(f"    ✗ Parameter test failed for {params}: {str(e)}")
            all_passed = False
    
    # Test 5: Stability detection
    print("\nTest 5: Stability detection")
    try:
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        # Reset and check stability
        substrate.reset_to_equilibrium(num_steps=1000)
        
        # Check if stability detection works
        is_stable = substrate.is_stable(window=50, threshold=0.001)
        print(f"    ✓ Stability detection: {is_stable}")
        
        # Should be stable after 1000 steps
        assert is_stable, "Substrate should be stable after 1000 steps"
        
    except Exception as e:
        print(f"    ✗ Stability detection failed: {str(e)}")
        all_passed = False
    
    # Test 6: Visualization
    print("\nTest 6: Visualization")
    try:
        substrate = GrayScottSubstrate(
            width=256, height=256,
            default_f=0.037, default_k=0.060,
            Du=0.16, Dv=0.08, dt=1.0,
            decay_rate=0.9995, device='cpu'
        )
        
        # Capture states during reset
        states = []
        states.append(substrate.V.clone().cpu().numpy())  # Initial
        
        substrate.reset_to_equilibrium(num_steps=500)
        states.append(substrate.V.clone().cpu().numpy())  # After reset
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        titles = ["Initial State", "After Equilibrium Reset"]
        for idx, (state, title) in enumerate(zip(states, titles)):
            im = axes[idx].imshow(state, cmap='viridis', origin='lower', vmin=0, vmax=1)
            axes[idx].set_title(title)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        viz_path = output_dir / 'equilibrium_reset.png'
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
            
            v_std = substrate.V.std().item()
            assert v_std > 0.01, f"No structure on device {device}"
            
            print(f"    ✓ Device {device}: V std={v_std:.4f}")
            
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
        
        start_time = time.time()
        substrate.reset_to_equilibrium(num_steps=500)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"    ✓ Time for 500 steps: {elapsed_time:.3f}s")
        
        # Should be reasonably fast
        assert elapsed_time < 5.0, f"Reset too slow: {elapsed_time:.3f}s"
        
    except Exception as e:
        print(f"    ✗ Performance test failed: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ EQUILIBRIUM RESET TEST: ALL PASSED")
        print("Equilibrium reset component is working correctly!")
    else:
        print("✗✗✗ EQUILIBRIUM RESET TEST: SOME FAILED")
        print("Check the errors above before running the full experiment.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = test_equilibrium_reset()
    sys.exit(0 if success else 1)