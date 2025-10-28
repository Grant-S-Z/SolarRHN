# Core Package Structure

The `core/` package contains modular, well-organized physics calculations for Solar RHN research.

## Module Organization

```
core/
├── __init__.py                         # Package initialization, exports all public functions
├── constants.py                        # Physical constants (π, GF, me, ℏ, s2w, etc.)
├── rhn_physics.py                      # RHN decay widths, lifetimes, spectra
├── transformations.py                  # Lorentz boosts & angle transformations
├── decay_distributions.py              # Decay kinematics (CMS & lab frames)
├── spectrum_utils.py                   # Integration, interpolation, file I/O
├── neutrino_electron_scattering.py     # ν-e scattering cross sections & spectra
├── decay_and_scattering.py             # RHN decay → ν scattering with azimuthal sampling
├── sampling.py                         # Monte Carlo rejection sampling algorithms
└── tools.py                            # Utility tools (timer decorator, debugging)
```

## Usage

### Import Everything
```python
from core import *
```

This imports all functions while maintaining clean namespaces.

### Import Specific Modules
```python
from core.constants import pi, GFermi, m_electron
from core.rhn_physics import RHN_TauCM, getRHNSpectrum
from core.transformations import cms_to_lab, transform_phi_to_theta
from core.decay_distributions import diff_El_costheta_lab
from core.spectrum_utils import integrateSpectrum2D
from core.neutrino_electron_scattering import scatter_electron_spectrum, mswlma
from core.decay_and_scattering import get_and_save_nuL_scatter_electron_El_costheta
from core.sampling import rejection_sampling_2Dfunc, getNuLEAndAngleBySampling
from core.tools import timer
```

## Module Details

### constants.py

Physical constants used throughout calculations:

- `pi`: 3.14159...
- `GFermi`: Fermi constant (MeV⁻²)
- `m_electron`: Electron mass (MeV)
- `hbar`: Reduced Planck constant (MeV·s)
- `s2w`: sin²θ_W (weak mixing angle)
- `distance_SE`: Sun-Earth distance (m)
- `speed_of_light`: c (m/s)

### rhn_physics.py

RHN decay physics:

- `RHN_Gamma_vvv(MH, U2)`: N → ννν decay width
- `RHN_Gamma_vll(MH, U2)`: N → νℓ⁺ℓ⁻ decay width
- `RHN_TauCM(MH, U2)`: RHN lifetime in CMS
- `RHN_BR_vll(MH, U2)`: Branching ratio to charged leptons
- `RHN_TauF(MH, EH)`: Time of flight for RHN
- `getRHNSpectrum(spectrum_L, MH, U2)`: Convert νL → N spectrum
- `getDecayedRHNSpectrum(...)`: Account for decay in flight
- `findRatioForDistance(MH, EH, U2, distance)`: Decay fraction
- `findDistanceForRatio(MH, EH, U2, ratio)`: Distance for target decay fraction

### transformations.py
Coordinate transformations:
- `cms_to_lab(El, costheta, MH, EH)`: Lorentz boost CMS → lab
- `lab_to_cms(El, costheta, MH, EH)`: Lorentz boost lab → CMS
- `transform_phi_to_theta(cosphi, distance)`: Sun-Earth frame → lab frame
- `transform_theta_to_phi(costheta, distance)`: Lab frame → Sun-Earth frame

### decay_distributions.py
Differential distributions:
- `diff_lambda(x, y, z)`: Phase space helper function
- `diff_El_costheta_cms(El, costheta, MH, EH)`: 2D distribution in CMS
- `diff_El_costheta_lab(El, costheta, MH, EH)`: 2D distribution in lab (with Jacobian)
- `diff_costheta(costheta, MH, EH)`: 1D angular distribution
- `diff_El(El, MH, EH)`: 1D energy distribution
- `diff_Eee(Eee, MH, EH)`: e⁺e⁻ pair energy distribution

### spectrum_utils.py
Spectrum operations:
- `interpolateSpectrum(inputCSV, xvalues)`: Interpolate from CSV
- `integrateSpectrum(spectrum)`: 1D trapezoidal integration
- `integrateSpectrum2D(data, ...)`: 2D integration with bin-aware algorithm
- `saveSpectrums(spectrums, column_names, fileName, labels)`: Save multiple spectra

### electron_scattering.py
Main physics calculations:
- `getNulEAndAngleFromRHNDecay(...)`: Core decay-in-flight calculation
- `get_and_save_nuL_El_costheta_decay_in_flight(...)`: Complete neutrino distributions
- `get_and_save_nuL_scatter_electron_El_costheta(...)`: Electron scattering with **azimuthal sampling** (12 φ points)
- `get_and_save_nuL_scatter_electron_El_costheta_from_csv(...)`: Process saved neutrino CSV

**Key Feature**: Azimuthal angle sampling for accurate angle mapping:
```
cos(θ_lab) = cos(θ_in)·cos(θ_s) - sin(θ_in)·sin(θ_s)·cos(φ)
```
Samples 12 azimuthal angles φ ∈ [0, 2π) for each (incoming angle, scatter angle) pair.

### neutrino_electron_scattering.py
Core ν-e elastic scattering physics (from solar.py):
- `scatter_electron_spectrum(energy, flux, ...)`: Compute 2D electron spectrum (energy × angle)
- `mswlma(energy)`: MSW-LMA oscillation probability P(νₑ → νₑ)
- `cal_Tmax(q)`: Maximum electron recoil energy
- `cal_costheta(T, q)`: Scattering angle from recoil energy
- Constants: `gL_e`, `gR_e`, `me`, `Ne`, `sig0_es`, etc.
- Binning parameters: `bin_width`, `max_energy`, `bin_array`, `bin_mid_array`

### sampling.py
Monte Carlo sampling algorithms:
- `rejection_sampling_2Dfunc(f, ...)`: 2D rejection sampling
- `rejection_sampling_1Dfunc(f, ...)`: 1D rejection sampling
- `generate_samples_from_spectrum(spectrum, ...)`: Sample from histogram
- `getMaximumValue2D(f, ...)`: Find function maximum on grid
- `getNuLEAndAngleBySampling(...)`: Generate neutrino samples from RHN decay (alternative to analytical method)

### tools.py

Utility tools for development and debugging:

- `timer`: Decorator to measure and print function execution time
  ```python
  @timer
  def my_function():
      # Function code here
      pass
  # Output: func: 'my_function' took: 1.2345 secs
  ```

## Backward Compatibility

The original `utils.py` file remains unchanged. All existing scripts continue to work.
The new `core/` package provides:
- ✓ Better organization
- ✓ Clear separation of concerns
- ✓ Easier testing and debugging
- ✓ Improved documentation
- ✓ No breaking changes

## Key Physics Features

### Normalization Strategy
2D distributions are stored as densities (per unit energy per unit angle).
Integration uses bin widths automatically:
```python
∫∫ ρ(E,θ) dE dθ = Σ ρ[i,j] × ΔE[i] × Δθ[j]
```
Normalization uses **simple sum** (not weighted) in denominator for consistency.

### Angle Mapping Improvements
1. **Jacobian transformation**: Proper dcosphi/dcostheta for φ ↔ θ conversion
2. **Direct evaluation**: Compute distribution at cosphi_needed, not via Jacobian bins
3. **Azimuthal sampling**: 12 φ points for accurate 3D → 2D projection

### Decay-in-Flight Integration
Divides Sun-Earth path into:
- 100 steps inside Earth orbit (distance < 1 AU)
- Adaptive steps outside Earth orbit (based on decay fraction)

Each step accumulates contributions to 2D and 1D distributions.

## Testing

To verify the refactoring didn't break anything:
```python
# Old way (still works)
from utils import *

# New way (cleaner)
from core import *

# Both should give identical results
```

## Future Improvements

Potential enhancements:
- [ ] Add type hints throughout
- [ ] Create unit tests for each module
- [ ] Parallelize batch processing
- [ ] Add progress bars for long computations
- [ ] Optimize energy resampling in scatter calculations
- [ ] Add caching for frequently-used calculations

## Contact

For questions about the code structure, see the main README.md.
