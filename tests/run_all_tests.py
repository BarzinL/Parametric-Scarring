#!/usr/bin/env python3
"""
Master Test Script for All Components

This script runs all component tests to verify the system is working correctly
before running the full experiment.

Usage:
    python tests/run_all_tests.py

Remember to activate the virtual environment first:
    source .venv/bin/activate
"""

import sys
import subprocess
import time
from pathlib import Path
import importlib.util

def run_test_script(script_name):
    """Run a test script and return success status"""
    print(f"\n{'='*80}")
    print(f"RUNNING TEST: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"\n✓✓✓ {script_name} PASSED (took {elapsed_time:.1f}s)")
            return True
        else:
            print(f"\n✗✗✗ {script_name} FAILED (took {elapsed_time:.1f}s)")
            print(f"Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗✗✗ {script_name} TIMEOUT (after 300s)")
        return False
    except Exception as e:
        print(f"\n✗✗✗ {script_name} ERROR: {str(e)}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_modules = [
        'torch', 'numpy', 'matplotlib', 'soundfile', 'scipy'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (missing)")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n✗✗✗ Missing dependencies: {', '.join(missing_modules)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("  ✓ All dependencies available")
    return True

def check_core_modules():
    """Check if core modules can be imported"""
    print("\nChecking core modules...")
    
    core_modules = [
        'core.substrate',
        'core.patterns',
        'core.scarring',
        'core.metrics',
        'core.visualization'
    ]
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    missing_modules = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} (import error: {e})")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n✗✗✗ Cannot import core modules: {', '.join(missing_modules)}")
        return False
    
    print("  ✓ All core modules importable")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("Checking virtual environment...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("  ✓ Running in virtual environment")
        return True
    else:
        print("  ⚠ Not running in virtual environment")
        print("    Consider activating with: source .venv/bin/activate")
        # Don't fail the test, but warn the user
        return True

def main():
    """Run all component tests"""
    print("="*80)
    print("COMPONENT TEST SUITE FOR PARAMETRIC SCARRING EXPERIMENT")
    print("="*80)
    print("\nThis script tests all components before running the full experiment.")
    print("Make sure you've activated the virtual environment first!\n")
    
    # Track test results
    test_results = {}
    
    # Pre-flight checks
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECKS")
    print("="*80)
    
    checks = [
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Core Modules", check_core_modules),
    ]
    
    all_checks_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_checks_passed = False
    
    if not all_checks_passed:
        print("\n✗✗✗ PRE-FLIGHT CHECKS FAILED")
        print("Please fix the issues above before running tests.")
        return False
    
    print("\n✓✓✓ ALL PRE-FLIGHT CHECKS PASSED")
    
    # List of test scripts to run
    test_scripts = [
        "test_audio_synthesis.py",
        "test_circular_mask.py", 
        "test_equilibrium_reset.py",
        "test_audio_injection.py"
    ]
    
    # Run each test
    total_start_time = time.time()
    
    for script in test_scripts:
        test_results[script] = run_test_script(script)
    
    total_elapsed_time = time.time() - total_start_time
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    for script, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {script}: {status}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print(f"Total time: {total_elapsed_time:.1f}s")
    
    # Final verdict
    print("\n" + "="*80)
    if passed_count == total_count:
        print("✓✓✓ ALL COMPONENT TESTS PASSED")
        print("\nThe system is ready to run the full experiment!")
        print("You can now run: python experiments/1a_rev1_direct_acoustic.py")
    else:
        print("✗✗✗ SOME COMPONENT TESTS FAILED")
        print("\nPlease fix the failing components before running the full experiment.")
        print("Check the error messages above for details.")
    print("="*80)
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)