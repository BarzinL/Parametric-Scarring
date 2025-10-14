#!/usr/bin/env python3
"""
Test Script for Circular Mask Component

This script tests the circular region creation functionality to ensure it works correctly
before running the full experiment.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.patterns import create_circular_region

def test_circular_mask():
    """Test circular mask creation functionality"""
    print("=" * 70)
    print("TESTING CIRCULAR MASK COMPONENT")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("test_results/circular_mask")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passed = True
    
    # Test 1: Basic circular mask creation
    print("\nTest 1: Basic circular mask creation")
    try:
        center = (128, 128)
        radius = 20
        grid_size = (256, 256)
        device = 'cpu'
        
        mask = create_circular_region(center, radius, grid_size, device)
        
        # Validate basic properties
        assert isinstance(mask, torch.Tensor), f"Mask is not a tensor: {type(mask)}"
        assert mask.shape == grid_size, f"Mask shape mismatch: {mask.shape} != {grid_size}"
        assert mask.dtype == torch.bool, f"Mask dtype incorrect: {mask.dtype}"
        assert mask.device.type == device, f"Device mismatch: {mask.device.type} != {device}"
        
        # Check that mask is boolean
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2, f"Mask has more than 2 unique values: {unique_values}"
        assert all(v in [torch.tensor(False), torch.tensor(True)] for v in unique_values), "Mask contains non-boolean values"
        
        # Count pixels in mask
        mask_count = mask.sum().item()
        expected_area = np.pi * radius * radius
        print(f"    ✓ Mask shape: {mask.shape}")
        print(f"    ✓ Pixels in mask: {mask_count}")
        print(f"    ✓ Expected area: {expected_area:.1f}")
        print(f"    ✓ Error: {abs(mask_count - expected_area) / expected_area * 100:.1f}%")
        
        # Visualize mask
        plt.figure(figsize=(6, 6))
        plt.imshow(mask.cpu().numpy(), cmap='gray', origin='lower')
        plt.title(f'Basic Circular Mask\nCenter: {center}, Radius: {radius}')
        plt.colorbar(label='Mask Value')
        plt.savefig(output_dir / 'basic_mask.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Visualization saved: {output_dir / 'basic_mask.png'}")
        
    except Exception as e:
        print(f"    ✗ Test failed: {str(e)}")
        all_passed = False
    
    # Test 2: Different positions
    print("\nTest 2: Different positions")
    positions = [(50, 50), (200, 100), (128, 200), (225, 225)]
    
    for center in positions:
        try:
            mask = create_circular_region(center, radius=20, grid_size=(256, 256), device='cpu')
            mask_count = mask.sum().item()
            
            # Basic checks
            assert mask_count > 0, f"Empty mask for center {center}"
            
            # Check center pixel is in mask
            cx, cy = center
            assert mask[cy, cx] == True, f"Center pixel not in mask for {center}"
            
            print(f"    ✓ Center {center}: {mask_count} pixels")
            
        except Exception as e:
            print(f"    ✗ Position test failed for {center}: {str(e)}")
            all_passed = False
    
    # Test 3: Different radii
    print("\nTest 3: Different radii")
    radii = [5, 10, 20, 50, 100]
    center = (128, 128)
    
    for radius in radii:
        try:
            mask = create_circular_region(center, radius, grid_size=(256, 256), device='cpu')
            mask_count = mask.sum().item()
            expected_area = np.pi * radius * radius
            
            # Check area is reasonable
            error = abs(mask_count - expected_area) / expected_area
            assert error < 0.1, f"Area error too large for radius {radius}: {error:.3f}"
            
            print(f"    ✓ Radius {radius}: {mask_count} pixels (expected {expected_area:.1f})")
            
        except Exception as e:
            print(f"    ✗ Radius test failed for {radius}: {str(e)}")
            all_passed = False
    
    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    
    # Test 4a: Radius = 0
    try:
        mask = create_circular_region((128, 128), radius=0, grid_size=(256, 256), device='cpu')
        mask_count = mask.sum().item()
        assert mask_count == 1, f"Radius 0 should have exactly 1 pixel: {mask_count}"
        print(f"    ✓ Radius 0: {mask_count} pixel(s)")
    except Exception as e:
        print(f"    ✗ Radius 0 test failed: {str(e)}")
        all_passed = False
    
    # Test 4b: Very large radius (exceeds grid)
    try:
        mask = create_circular_region((128, 128), radius=500, grid_size=(256, 256), device='cpu')
        mask_count = mask.sum().item()
        total_pixels = 256 * 256
        assert mask_count == total_pixels, f"Large radius should cover all pixels: {mask_count}/{total_pixels}"
        print(f"    ✓ Large radius: {mask_count}/{total_pixels} pixels")
    except Exception as e:
        print(f"    ✗ Large radius test failed: {str(e)}")
        all_passed = False
    
    # Test 4c: Center at edges
    edge_centers = [(0, 0), (255, 0), (0, 255), (255, 255)]
    for center in edge_centers:
        try:
            mask = create_circular_region(center, radius=30, grid_size=(256, 256), device='cpu')
            mask_count = mask.sum().item()
            assert mask_count > 0, f"Edge center {center} produced empty mask"
            print(f"    ✓ Edge center {center}: {mask_count} pixels")
        except Exception as e:
            print(f"    ✗ Edge center test failed for {center}: {str(e)}")
            all_passed = False
    
    # Test 5: Different grid sizes
    print("\nTest 5: Different grid sizes")
    grid_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for grid_size in grid_sizes:
        try:
            center = (grid_size[1] // 2, grid_size[0] // 2)
            radius = min(grid_size) // 8
            mask = create_circular_region(center, radius, grid_size, device='cpu')
            
            assert mask.shape == grid_size, f"Shape mismatch for {grid_size}: {mask.shape}"
            mask_count = mask.sum().item()
            assert mask_count > 0, f"Empty mask for grid size {grid_size}"
            
            print(f"    ✓ Grid {grid_size}: {mask_count} pixels")
            
        except Exception as e:
            print(f"    ✗ Grid size test failed for {grid_size}: {str(e)}")
            all_passed = False
    
    # Test 6: Device compatibility
    print("\nTest 6: Device compatibility")
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        try:
            mask = create_circular_region((128, 128), radius=20, grid_size=(256, 256), device=device)
            assert mask.device.type == device, f"Device mismatch: {mask.device.type} != {device}"
            print(f"    ✓ Device {device}: {mask.shape}")
        except Exception as e:
            print(f"    ✗ Device test failed for {device}: {str(e)}")
            all_passed = False
    
    # Test 7: Visualization of multiple masks
    print("\nTest 7: Visualization of multiple masks")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Different radii
        for idx, radius in enumerate([10, 20, 40]):
            mask = create_circular_region((128, 128), radius, (256, 256), 'cpu')
            axes[0, idx].imshow(mask.cpu().numpy(), cmap='gray', origin='lower')
            axes[0, idx].set_title(f'Radius: {radius}')
            axes[0, idx].axis('off')
        
        # Different positions
        centers = [(64, 64), (192, 64), (128, 192)]
        for idx, center in enumerate(centers):
            mask = create_circular_region(center, 20, (256, 256), 'cpu')
            axes[1, idx].imshow(mask.cpu().numpy(), cmap='gray', origin='lower')
            axes[1, idx].set_title(f'Center: {center}')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        viz_path = output_dir / 'mask_variations.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"    ✗ Visualization failed: {str(e)}")
        all_passed = False
    
    # Test 8: Performance test
    print("\nTest 8: Performance test")
    try:
        import time
        
        start_time = time.time()
        for _ in range(100):
            mask = create_circular_region((128, 128), radius=20, grid_size=(256, 256), device='cpu')
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        print(f"    ✓ Average time per mask: {avg_time:.2f}ms")
        
        # Should be reasonably fast
        assert avg_time < 10, f"Mask creation too slow: {avg_time:.2f}ms"
        
    except Exception as e:
        print(f"    ✗ Performance test failed: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ CIRCULAR MASK TEST: ALL PASSED")
        print("Circular mask component is working correctly!")
    else:
        print("✗✗✗ CIRCULAR MASK TEST: SOME FAILED")
        print("Check the errors above before running the full experiment.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = test_circular_mask()
    sys.exit(0 if success else 1)