# Egent: LLM-Powered Equivalent Width Measurement

An autonomous equivalent width (EW) measurement system for high-resolution stellar spectra, combining optimized Voigt fitting with LLM vision-based quality assessment.

**🌐 Try it online: [https://ew-agent.streamlit.app/](https://ew-agent.streamlit.app/)**

## Overview

Egent uses a **two-stage approach**:

1. **Direct Voigt Fitting**: Fast, deterministic multi-Voigt fitting with automated continuum estimation and quality metrics.

2. **LLM Visual Inspection**: For borderline cases, the LLM visually inspects fit plots and can:
   - Adjust the extraction window
   - Add blend components for missed lines
   - Modify continuum treatment
   - Flag unreliable measurements

### Key Features

- **Simple Input**: Just provide a spectrum file and a line list
- **Rest-Frame Input**: Expects spectra already in the stellar rest frame
- **Complete Provenance**: All Voigt parameters, continuum coefficients, and LLM reasoning stored
- **Parallel Processing**: Thread pool execution for high throughput

> **Coming Soon**: Fully offline version using [Ollama](https://ollama.ai/) with Qwen3-VL-8B for local inference without API costs.

## Installation

```bash
pip install numpy pandas scipy matplotlib openai python-dotenv streamlit
```

## Configuration

Add your OpenAI API key to your environment:

```bash
export OPENAI_API_KEY='your-openai-key'
```

Or create a `~/.env` file:
```bash
OPENAI_API_KEY=your-openai-key
```

Optional: Override the default model (GPT-5-mini):
```bash
export EGENT_MODEL=gpt-5
```

## Quick Start

### Command Line

```bash
# Run on example data
python run_ew.py --spectrum example/spectrum.csv --lines example/linelist.csv

# With custom output directory
python run_ew.py --spectrum example/spectrum.csv --lines example/linelist.csv --output-dir ./output

# Adjust number of parallel workers
python run_ew.py --spectrum example/spectrum.csv --lines example/linelist.csv --workers 5
```

### Web Interface (Streamlit)

**Try it online:** [https://ew-agent.streamlit.app/](https://ew-agent.streamlit.app/)

Or run locally:
```bash
streamlit run app.py
```

The web app allows you to:
- Upload spectrum and line list files (or use example data)
- Enter your OpenAI API key
- Watch results and plots update in real-time
- Download results as CSV, JSON, or ZIP of plots

## Tutorial

For a detailed walkthrough with explanations, see **`tutorial.ipynb`**. The Jupyter notebook covers:
- Setting up your OpenAI API key
- Understanding the input data format
- The physics of Voigt profile fitting
- Running the pipeline and interpreting results
- Reading the output JSON and diagnostic plots

The tutorial is self-contained and installs all dependencies automatically.

## Input File Formats

### Spectrum File (CSV)

The spectrum must be in the stellar rest frame with three columns:

```csv
wavelength,flux,flux_error
6100.00,12500.5,125.0
6100.05,12480.2,124.8
6100.10,12495.1,124.9
...
```

- `wavelength`: Wavelength in Angstroms (rest frame)
- `flux`: Flux values (any units)
- `flux_error`: Flux uncertainty (same units as flux)

### Line List File (CSV)

A simple list of wavelengths to measure:

```csv
wavelength
6125.03
6137.69
6142.49
6145.02
```

## Output

### Results JSON

Results are saved as JSON with complete metadata:

```
~/Egent_output/results_<timestamp>.json
```

Contains:
- `metadata`: Run configuration
- `summary`: Statistics (direct/llm/flagged/timeout counts)
- `results`: Per-line measurements including:
  - Measured EW and uncertainty
  - Direct vs LLM-refined values
  - Full Voigt parameters for exact reconstruction
  - LLM conversation logs (when triggered)

### Diagnostic Plots

```
~/Egent_output/fits/
├── direct/     # Direct fits (no LLM needed)
├── llm/        # LLM-improved fits
├── flagged/    # Flagged lines (unreliable)
└── error/      # Lines with errors
```

Each plot shows:
- Normalized flux with Voigt model overlay
- Individual line centers marked
- Residuals with σ-bands
- RMS quality metric

## How It Works

### Stage 1: Direct Fitting

For each line in the line list:

1. Extract ±3 Å region around target wavelength
2. Estimate continuum using iterative sigma-clipping
3. Find absorption peaks in inverted normalized flux
4. Fit multi-Voigt model (polynomial continuum + N Voigt profiles)
5. Compute quality metrics (RMS, χ², residual slope)

### Stage 2: LLM Review (if needed)

Lines trigger LLM inspection if:
- Quality is "poor" (high RMS, bad χ²)
- Region is crowded (>10 lines)
- Residual slope indicates continuum problems

The LLM receives the diagnostic plot and can:
- Call `extract_region()` to adjust window size
- Call `set_continuum_method()` to try polynomial continuum
- Call `set_continuum_regions()` for manual continuum selection
- Call `fit_ew(additional_peaks=[...])` to add blend components
- Call `flag_line()` to mark unreliable measurements
- Call `record_measurement()` to accept the fit

### Quality Thresholds

- **RMS < 1.5σ**: Excellent fit → auto-accept
- **RMS 1.5-2.0σ**: Good fit → accept if target region clean
- **RMS 2.0-2.5σ**: Marginal → LLM review
- **RMS > 2.5σ**: Poor → LLM must improve or flag
- **RMS > 3.0σ**: Auto-flagged as unreliable

## Directory Structure

```
Egent/
├── app.py             # Streamlit web interface
├── config.py          # Configuration (API key, model)
├── llm_client.py      # OpenAI client with retry logic
├── ew_tools.py        # Core EW measurement functions (LLM tools)
├── run_ew.py          # Main analysis pipeline (CLI)
├── tutorial.ipynb     # Step-by-step tutorial notebook
├── example/           # Example data files
│   ├── spectrum.csv   # Sample high-SNR Magellan/MIKE spectrum
│   └── linelist.csv   # Sample line list (14 lines)
└── README.md
```

## API Reference

### LLM Tools (in `ew_tools.py`)

| Function | Description |
|----------|-------------|
| `load_spectrum(file)` | Load rest-frame spectrum from CSV |
| `extract_region(wave, window)` | Extract spectral region around target |
| `set_continuum_method(method, order)` | Configure continuum fitting |
| `set_continuum_regions(regions)` | Manual continuum specification |
| `fit_ew(additional_peaks=[])` | Multi-Voigt fitting with blend support |
| `get_fit_plot()` | Generate diagnostic plot for inspection |
| `flag_line(wave, reason)` | Flag line as unreliable |
| `record_measurement(...)` | Record final EW measurement |

## Example Session

```python
from ew_tools import load_spectrum, extract_region, fit_ew, get_fit_plot

# Load spectrum
load_spectrum('example/spectrum.csv')

# Extract region around a line
extract_region(6142.49, window=3.0)

# Fit and get results
result = fit_ew()
print(f"EW = {result['target_line']['ew_mA']:.1f} mÅ")
print(f"Quality: {result['fit_quality']}")

# Generate plot
plot = get_fit_plot()
# plot['image_base64'] contains the PNG image
```

## Notes

- **Rest Frame**: The spectrum must already be shifted to the stellar rest frame. Apply barycentric and radial velocity corrections before using Egent.
- **Wavelength Units**: All wavelengths are in Angstroms.
- **EW Units**: Equivalent widths are reported in milli-Angstroms (mÅ).

## Citation

If you use Egent in your research, please cite:

```
Ting et al. (2025), "Egent: An Autonomous Agent for Equivalent Width Measurement"
```

## License

MIT License
