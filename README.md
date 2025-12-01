# Egent: LLM-Powered Equivalent Width Measurement Agent

An autonomous equivalent width (EW) measurement system for high-resolution stellar spectra, combining optimized Voigt fitting with LLM vision-based quality assessment.

## Overview

Egent uses a **two-stage approach**:

1. **Direct Voigt Fitting**: Fast, deterministic fitting with optimized continuum estimation. Automatically accepts fits passing quality thresholds.

2. **LLM Visual Inspection**: For borderline cases, LLM vision capability visually inspects fit plots and can adjust continuum, window size, add blend peaks, or flag unreliable lines.

### Key Design Principles

- **Blind Analysis**: No catalog EW values are used for fitting decisions (only for final evaluation)
- **Conservative**: LLM intervention only when diagnostics indicate uncertainty  
- **Parallel Processing**: Thread pool execution for high throughput
- **Complete Provenance**: All fit iterations, Voigt parameters, and LLM reasoning stored

## Installation

```bash
pip install numpy pandas scipy matplotlib openai python-dotenv
```

## Configuration

Add your API key to `~/.env`:

```bash
# For Azure OpenAI (default)
AZURE_API_KEY=your-azure-key
# or
AZURE=your-azure-key

# For OpenAI API (optional)
OPENAI_API_KEY=your-openai-key

# Select backend (optional, auto-detected from available keys)
EGENT_BACKEND=azure  # or 'openai'

# Override model (optional, defaults: 'gpt-5' or 'gpt-5-mini')
EGENT_MODEL=gpt-5
```

## Quick Start

```bash
# Run analysis using Azure OpenAI (default)
python run_ew.py --gaia-id 4287378367995943680

# Use mini model (faster, cheaper)
python run_ew.py --gaia-id 4287378367995943680 --mini

# Limit lines for testing
python run_ew.py --gaia-id 4287378367995943680 --n-lines 20

# Light elements only (Z < 26)
python run_ew.py --gaia-id 4287378367995943680 --light-only

# Using OpenAI backend
EGENT_BACKEND=openai python run_ew.py --gaia-id 4287378367995943680
```

## Directory Structure

```
Egent_Development/
├── config.py               # Centralized configuration (API backends, models)
├── llm_client.py           # Unified LLM client with retry logic
├── ew_tools.py             # Core EW measurement functions (LLM tools)
├── utils.py                # Utility functions (plotting, helpers)
├── run_ew.py               # Main EW analysis pipeline
├── run_all_spectra.py      # Batch processing for multiple spectra
├── preprocess_spectra.py   # Pre-apply wavelength corrections
├── replot_results.py       # Regenerate plots from saved results
├── combine_results.py      # Combine individual results into aggregate format
├── data/                   # Catalog and calibration data (not in repo)
│   ├── c3po_equivalent_widths.csv      # Line catalog with reference EWs
│   ├── magellan_bary_corrections.csv   # Barycentric corrections (for raw spectra)
│   └── spectra_good_calibration.csv    # Calibration info for good spectra
├── spectra/                # Raw Magellan spectra (not in repo)
├── spectra_corrected/      # Pre-corrected spectra - PREFERRED (not in repo)
└── ~/Egent_output/         # Default output directory (or ~/Egent_output_mini/)
```

## Core Module: `ew_tools.py`

Thread-safe functions for multi-step fitting workflows:

| Function | Description |
|----------|-------------|
| `load_spectrum(gaia_id)` | Load pre-corrected spectrum (wavelengths in rest frame) |
| `extract_region(wavelength, window)` | Extract spectral region around target |
| `set_continuum_method(method, order)` | Configure continuum (linear default, polynomial for curved) |
| `set_continuum_regions(regions)` | Manual continuum specification for crowded regions |
| `fit_ew(additional_peaks=[])` | Multi-Voigt fitting; can add peaks for blends |
| `get_fit_plot()` | Generate diagnostic plot for visual inspection |
| `flag_line(wavelength, reason)` | Flag line as unreliable |
| `record_measurement(...)` | Record final EW measurement |

## Output

Results saved as JSON with complete metadata:
```
~/Egent_output/results_gaia{ID}_{timestamp}.json
```

Contains:
- `metadata`: Run configuration (gaia_id, pair_id, component, model, backend, timestamp, n_lines, n_workers)
- `summary`: Statistics (direct/llm/flagged/timeout counts)
- `results`: Per-line measurements with:
  - Direct EW from automated fitting
  - Final measured EW (direct or LLM-refined)
  - Full diagnostics (χ², RMS, n_lines, etc.)
  - All Voigt parameters for exact reconstruction
  - LLM conversation logs and reasoning (when LLM was used)

### Generated Plots

| Plot | Description |
|------|-------------|
| `gaia{ID}_fits/direct/*.png` | Direct fits (no LLM needed) |
| `gaia{ID}_fits/llm/*.png` | LLM-improved fits |
| `gaia{ID}_fits/flagged/*.png` | Flagged lines (unreliable measurements) |
| `gaia{ID}_fits/error/*.png` | Lines with errors (no data, timeouts, etc.) |
| `*_comparison_*.png` | 1-to-1 catalog vs measured with LLM correction arrows |

To regenerate plots from saved results:
```bash
python replot_results.py output/results_gaia*.json --all-lines
```

## Batch Processing

```bash
# Process all well-calibrated spectra
python run_all_spectra.py

# Use mini model
python run_all_spectra.py --mini

# Resume interrupted run
python run_all_spectra.py --resume

# Preview without running
python run_all_spectra.py --dry-run

# Include poorly calibrated spectra
python run_all_spectra.py --all-spectra

# Custom worker count
python run_all_spectra.py --workers 20
```

## Command Line Options

### `run_ew.py`

```
--gaia-id ID          Gaia DR3 source ID (required)
--workers N           Number of parallel workers
--light-only          Only analyze light elements (Z < 26)
--custom-spectrum     Path to custom spectrum file
--n-lines N           Limit number of lines (for testing)
--output-dir DIR      Custom output directory
--mini                Use mini model (faster, cheaper)
```

### `run_all_spectra.py`

```
--workers N           Parallel workers
--resume              Skip already processed spectra
--dry-run             List spectra without processing
--limit N             Limit number of spectra
--n-lines N           Limit lines per spectrum
--all-spectra         Include poorly calibrated spectra
--output-dir DIR      Custom output directory
--mini                Use mini model
```

## Wavelength Calibration

**Pre-corrected spectra (required)**: Spectra must be pre-processed and placed in `spectra_corrected/` with barycentric + empirical corrections already applied. The pipeline loads these corrected spectra directly - no runtime calibration is performed.

**Pre-processing workflow**:
1. Barycentric correction from `magellan_bary_corrections.csv`
2. Empirical calibration using Fe I lines
3. Save to `spectra_corrected/` for analysis

To pre-process spectra:
```bash
# Process all good spectra
python preprocess_spectra.py

# Process single spectrum
python preprocess_spectra.py --gaia-id 1234567890

# Preview without saving
python preprocess_spectra.py --dry-run
```

## Fit Quality Diagnostics

The fitting pipeline computes rich diagnostics:

- **Normalized RMS**: Residuals divided by uncertainty (expect ~1.0)
- **Reduced χ²**: Chi-squared per degree of freedom
- **Central RMS**: Focused on ±1.5Å around target (most important)
- **Correlated residuals**: Consecutive high-residual points (indicates blending/bad continuum)
- **Residual slope**: Linear trend in normalized residuals (σ/Å) - signals continuum errors

### LLM Intervention Triggers
- Poor fit quality assessment
- 10+ lines in window (very crowded)
- χ² > 15 or central RMS > 2.5
- Residual slope > 0.5 σ/Å (continuum problem)

### Auto-Flagging Criteria
- RMS > 3.0σ → Automatically flagged by pipeline (safety net)
- Lines flagged during fitting stage are excluded from statistics

## API Backends

Egent supports both OpenAI and Azure OpenAI:

```python
from config import get_config

cfg = get_config()
print(f"Backend: {cfg.backend}")  # 'openai' or 'azure'
print(f"Model: {cfg.model_id}")   # 'gpt-5' (default) or 'gpt-5-mini'
```

Backend defaults to 'azure' but is auto-detected from available API keys, or set explicitly:
```bash
export EGENT_BACKEND=azure  # or 'openai'
```

## Author

Yuan-Sen Ting (OSU)

## License

MIT License
