"""
Quick test to verify core package imports work correctly.

This script tests that all modules can be imported and key functions are accessible.
"""

def test_imports():
    """Test that all core modules import correctly."""
    print("Testing core package imports...\n")
    
    # Test 1: Import package
    print("1. Testing package import...")
    try:
        import core
        print("   ✓ core package imported")
    except Exception as e:
        print(f"   ✗ Failed to import core: {e}")
        return False
    
    # Test 2: Import constants
    print("\n2. Testing constants module...")
    try:
        from core.constants import pi, GFermi, m_electron, distance_SE
        print(f"   ✓ Constants imported: π={pi:.4f}, GF={GFermi:.2e}, me={m_electron:.4f} MeV")
    except Exception as e:
        print(f"   ✗ Failed to import constants: {e}")
        return False
    
    # Test 3: Import RHN physics
    print("\n3. Testing rhn_physics module...")
    try:
        from core.rhn_physics import RHN_TauCM, getRHNSpectrum
        print(f"   ✓ RHN physics functions imported")
        # Quick sanity check
        tau = RHN_TauCM(4.0, 1e-6)
        print(f"   Example: τ_CM(M=4 MeV, U²=10⁻⁶) = {tau:.2e} s")
    except Exception as e:
        print(f"   ✗ Failed to import rhn_physics: {e}")
        return False
    
    # Test 4: Import transformations
    print("\n4. Testing transformations module...")
    try:
        from core.transformations import cms_to_lab, transform_phi_to_theta
        print("   ✓ Transformation functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import transformations: {e}")
        return False
    
    # Test 5: Import decay distributions
    print("\n5. Testing decay_distributions module...")
    try:
        from core.decay_distributions import diff_El_costheta_lab, diff_lambda
        print("   ✓ Decay distribution functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import decay_distributions: {e}")
        return False
    
    # Test 6: Import spectrum utilities
    print("\n6. Testing spectrum_utils module...")
    try:
        from core.spectrum_utils import integrateSpectrum, integrateSpectrum2D
        print("   ✓ Spectrum utility functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import spectrum_utils: {e}")
        return False
    
    # Test 7: Import electron scattering
    print("\n7. Testing electron_scattering module...")
    try:
        from core.electron_scattering import (
            getNulEAndAngleFromRHNDecay,
            get_and_save_nuL_El_costheta_decay_in_flight
        )
        print("   ✓ Electron scattering functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import electron_scattering: {e}")
        return False
    
    # Test 8: Import neutrino-electron scattering module
    print("\n8. Testing neutrino_electron_scattering module...")
    try:
        from core.neutrino_electron_scattering import scatter_electron_spectrum, mswlma, cal_Tmax
        print("   ✓ Neutrino-electron scattering functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import neutrino_electron_scattering: {e}")
        return False
    
    # Test 9: Import sampling module
    print("\n9. Testing sampling module...")
    try:
        from core.sampling import rejection_sampling_2Dfunc, generate_samples_from_spectrum
        print("   ✓ Sampling functions imported")
    except Exception as e:
        print(f"   ✗ Failed to import sampling: {e}")
        return False
    
    # Test 10: Import tools module
    print("\n10. Testing tools module...")
    try:
        from core.tools import timer
        print("   ✓ Tools imported (timer decorator)")
        # Test the timer decorator
        @timer
        def test_func():
            import time
            time.sleep(0.01)
            return "test"
        result = test_func()
        print("   ✓ Timer decorator works")
    except Exception as e:
        print(f"   ✗ Failed to import tools: {e}")
        return False
    
    # Test 11: Test that __all__ is properly defined
    print("\n11. Testing package __all__ attribute...")
    try:
        import core
        assert hasattr(core, '__all__'), "core package missing __all__"
        assert len(core.__all__) > 0, "__all__ is empty"
        print(f"   ✓ Package exports {len(core.__all__)} functions")
        # Check a few key functions are in __all__
        assert 'pi' in core.__all__
        assert 'getRHNSpectrum' in core.__all__
        assert 'diff_El_costheta_lab' in core.__all__
        assert 'scatter_electron_spectrum' in core.__all__
        assert 'timer' in core.__all__
        print("   ✓ Key functions in __all__")
    except Exception as e:
        print(f"   ✗ Failed __all__ test: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed! Core package is working correctly.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
