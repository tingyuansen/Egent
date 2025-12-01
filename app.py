#!/usr/bin/env python3
"""
Egent Streamlit App
===================

Web interface for LLM-powered equivalent width measurement.
Upload spectrum and line list, enter API key, get results with real-time plots.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
import io
import base64
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Egent - EW Measurement",
    page_icon="⭐",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text { color: #28a745; }
    .warning-text { color: #ffc107; }
    .error-text { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">⭐ Egent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LLM-Powered Equivalent Width Measurement</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your OpenAI API key (starts with 'sk-')"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        if api_key.startswith('sk-') and len(api_key) > 20:
            st.success("✓ API key set")
        else:
            st.warning("⚠ Invalid key format")
    
    st.markdown("---")
    
    # Processing options
    st.subheader("Processing Options")
    n_workers = st.slider("Parallel Workers", 1, 10, 3, 
                          help="More workers = faster, but may hit rate limits")
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload spectrum CSV
    2. Upload line list CSV
    3. Enter your API key
    4. Click 'Run Analysis'
    
    [📖 Documentation](https://github.com/tingyuansen/Egent)
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Spectrum File")
    spectrum_file = st.file_uploader(
        "Upload spectrum CSV",
        type=['csv'],
        help="CSV with columns: wavelength, flux, flux_error"
    )
    
    if spectrum_file:
        try:
            spectrum_df = pd.read_csv(spectrum_file)
            st.success(f"✓ Loaded {len(spectrum_df)} points")
            
            # Show preview
            with st.expander("Preview spectrum"):
                st.dataframe(spectrum_df.head(10))
                
                # Quick plot
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(spectrum_df['wavelength'], spectrum_df['flux'], 'k-', lw=0.5)
                ax.set_xlabel('Wavelength (Å)')
                ax.set_ylabel('Flux')
                ax.set_title(f"Spectrum: {spectrum_df['wavelength'].min():.1f} - {spectrum_df['wavelength'].max():.1f} Å")
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.error(f"Error loading spectrum: {e}")

with col2:
    st.subheader("📋 Line List")
    linelist_file = st.file_uploader(
        "Upload line list CSV",
        type=['csv'],
        help="CSV with column: wavelength"
    )
    
    if linelist_file:
        try:
            linelist_df = pd.read_csv(linelist_file)
            if 'wavelength' not in linelist_df.columns:
                linelist_df.columns = ['wavelength'] + list(linelist_df.columns[1:])
            st.success(f"✓ Loaded {len(linelist_df)} lines")
            
            # Show preview
            with st.expander("Preview line list"):
                st.dataframe(linelist_df)
        except Exception as e:
            st.error(f"Error loading line list: {e}")

st.markdown("---")

# Run button
can_run = (
    spectrum_file is not None and 
    linelist_file is not None and 
    api_key and api_key.startswith('sk-')
)

if st.button("🚀 Run Analysis", disabled=not can_run, type="primary"):
    if not can_run:
        st.error("Please upload both files and enter a valid API key")
    else:
        # Save uploaded files temporarily
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)
        
        spectrum_path = temp_dir / "spectrum.csv"
        linelist_path = temp_dir / "linelist.csv"
        
        # Reset file positions and save
        spectrum_file.seek(0)
        linelist_file.seek(0)
        
        with open(spectrum_path, 'wb') as f:
            f.write(spectrum_file.read())
        with open(linelist_path, 'wb') as f:
            f.write(linelist_file.read())
        
        # Import Egent modules
        try:
            from ew_tools import load_spectrum, extract_region, set_continuum_method, fit_ew, get_fit_plot, _get_session
            from run_ew import direct_fit, llm_measure_with_vision
            from config import get_config
            from scipy.special import voigt_profile
        except ImportError as e:
            st.error(f"Import error: {e}")
            st.stop()
        
        # Load data
        load_result = load_spectrum(str(spectrum_path))
        if not load_result['success']:
            st.error(f"Failed to load spectrum: {load_result.get('error')}")
            st.stop()
        
        linelist_df = pd.read_csv(linelist_path)
        if 'wavelength' not in linelist_df.columns:
            linelist_df.columns = ['wavelength'] + list(linelist_df.columns[1:])
        line_waves = linelist_df['wavelength'].dropna().values.tolist()
        
        st.info(f"Processing {len(line_waves)} lines with {n_workers} workers...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results containers
        results_container = st.container()
        
        # Create columns for results
        col_results, col_plots = st.columns([1, 2])
        
        with col_results:
            st.subheader("📈 Results")
            results_table = st.empty()
        
        with col_plots:
            st.subheader("🔬 Diagnostic Plots")
            plots_container = st.container()
        
        # Process lines
        results = []
        stats = {'direct': 0, 'llm': 0, 'flagged': 0}
        
        for i, line_wave in enumerate(line_waves):
            progress = (i + 1) / len(line_waves)
            progress_bar.progress(progress)
            status_text.text(f"Processing line {i+1}/{len(line_waves)}: {line_wave:.2f} Å")
            
            start_time = time.time()
            
            # Direct fit first
            direct_result = direct_fit(str(spectrum_path), line_wave)
            
            d_ew = direct_result.get('measured_ew')
            d_qual = direct_result.get('fit_quality', '?')
            needs_llm = direct_result.get('needs_improvement', False)
            
            result = {
                'wavelength': line_wave,
                'direct_ew': d_ew,
                'direct_quality': d_qual,
                'direct_voigt_params': direct_result.get('voigt_params'),
                'region_info': direct_result.get('region_info'),
                'continuum_info': direct_result.get('continuum_info'),
            }
            
            if direct_result.get('flagged'):
                result.update({
                    'success': False,
                    'flagged': True,
                    'flag_reason': direct_result.get('flag_reason', 'no_data'),
                    'method': 'direct',
                    'used_llm': False,
                    'time_sec': time.time() - start_time,
                })
                stats['flagged'] += 1
            elif direct_result.get('success') and not needs_llm:
                result.update({
                    'success': True,
                    'measured_ew': d_ew,
                    'ew_err': direct_result['ew_err'],
                    'fit_quality': d_qual,
                    'method': 'direct',
                    'used_llm': False,
                    'flagged': False,
                    'time_sec': time.time() - start_time,
                })
                stats['direct'] += 1
            else:
                # Need LLM
                status_text.text(f"Line {line_wave:.2f} Å: LLM reviewing ({direct_result.get('improvement_reason', 'quality')})")
                
                llm_result = llm_measure_with_vision(
                    str(spectrum_path), line_wave, direct_result, str(temp_dir)
                )
                
                final_region = llm_result.get('iterations', [{}])[-1].get('region_info') or direct_result.get('region_info')
                final_continuum = llm_result.get('iterations', [{}])[-1].get('continuum_info') or direct_result.get('continuum_info')
                
                result.update({
                    'success': llm_result.get('success', False),
                    'measured_ew': llm_result.get('measured_ew'),
                    'ew_err': llm_result.get('ew_err'),
                    'fit_quality': llm_result.get('fit_quality'),
                    'flagged': llm_result.get('flagged', False),
                    'flag_reason': llm_result.get('flag_reason'),
                    'method': 'llm',
                    'n_iterations': llm_result.get('n_iterations', 0),
                    'iterations': llm_result.get('iterations', []),
                    'used_llm': True,
                    'time_sec': time.time() - start_time,
                    'region_info': final_region,
                    'continuum_info': final_continuum,
                })
                
                if result.get('flagged'):
                    stats['flagged'] += 1
                else:
                    stats['llm'] += 1
            
            results.append(result)
            
            # Update results table
            results_df = pd.DataFrame([{
                'Wavelength': r['wavelength'],
                'EW (mÅ)': f"{r.get('measured_ew', 0):.1f}" if r.get('measured_ew') else "---",
                'Error': f"±{r.get('ew_err', 0):.1f}" if r.get('ew_err') else "---",
                'Quality': r.get('fit_quality', '---'),
                'Method': 'LLM' if r.get('used_llm') else 'Direct',
                'Status': '🚩' if r.get('flagged') else '✓',
            } for r in results])
            results_table.dataframe(results_df, use_container_width=True)
            
            # Generate and display plot
            with plots_container:
                try:
                    # Re-run fit to get plot data
                    load_spectrum(str(spectrum_path))
                    region_info = result.get('region_info') or {}
                    window = region_info.get('window', 3.0)
                    extract_region(line_wave, window=window)
                    
                    continuum_info = result.get('continuum_info') or {}
                    method = continuum_info.get('method', 'iterative_linear')
                    if 'poly' in method:
                        set_continuum_method('polynomial', order=2)
                    else:
                        set_continuum_method('iterative_linear', order=1)
                    
                    fit_result = fit_ew()
                    
                    if fit_result.get('success'):
                        session = _get_session()
                        last_fit = session['last_fit']
                        w = np.array(last_fit['wave'])
                        f_norm = np.array(last_fit['flux_norm'])
                        f_norm_err = np.array(last_fit['flux_norm_err'])
                        model_norm = np.array(last_fit['flux_fit'])
                        all_lines = last_fit['all_lines']
                        
                        residuals = f_norm - model_norm
                        residuals_norm = residuals / f_norm_err if np.any(f_norm_err > 0) else residuals / 0.01
                        rms = float(np.std(residuals_norm))
                        
                        # Create plot
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[3, 1], sharex=True)
                        
                        ax1.plot(w, f_norm, 'k-', lw=1.0, alpha=0.9, label='Data')
                        ax1.plot(w, model_norm, 'r-', lw=1.5, label='Fit')
                        ax1.axhline(1, color='gray', ls=':', alpha=0.5)
                        ax1.axvline(line_wave, color='blue', ls=':', lw=2, alpha=0.7, label='Target')
                        
                        for line in all_lines:
                            color = 'green' if abs(line['center'] - line_wave) < 0.3 else 'orange'
                            ax1.axvline(line['center'], color=color, ls='--', alpha=0.5, lw=1)
                        
                        meas_ew = result.get('measured_ew', 0)
                        flagged = result.get('flagged', False)
                        used_llm = result.get('used_llm', False)
                        status = "FLAGGED" if flagged else ("LLM" if used_llm else "DIRECT")
                        ax1.set_title(f'{line_wave:.2f} Å | {status} | EW={meas_ew:.1f} mÅ', fontsize=11,
                                     weight='bold', color='red' if flagged else 'black')
                        ax1.set_ylabel('Normalized Flux')
                        ax1.set_ylim(0.3, 1.15)
                        ax1.legend(loc='lower right', fontsize=9)
                        
                        ax2.axhspan(-1, 1, alpha=0.2, color='lightgreen')
                        ax2.axhspan(-2, 2, alpha=0.1, color='lightyellow')
                        ax2.plot(w, residuals_norm, 'k-', lw=0.8)
                        ax2.axhline(0, color='gray', ls='-', alpha=0.5)
                        ax2.set_xlabel('Wavelength (Å)')
                        ax2.set_ylabel('Residuals (σ)')
                        ax2.set_ylim(-4, 4)
                        ax2.text(0.02, 0.95, f'RMS={rms:.2f}σ', transform=ax2.transAxes, fontsize=10, va='top')
                        
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                except Exception as e:
                    st.warning(f"Could not generate plot for {line_wave:.2f} Å: {e}")
        
        # Final summary
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        st.markdown("---")
        st.subheader("📊 Summary")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Lines", len(results))
        col_s2.metric("Direct Fits", stats['direct'])
        col_s3.metric("LLM Refined", stats['llm'])
        col_s4.metric("Flagged", stats['flagged'])
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = {
            'metadata': {
                'timestamp': timestamp,
                'n_lines': len(line_waves),
                'n_workers': n_workers,
            },
            'summary': stats,
            'results': results,
        }
        
        results_json = json.dumps(output, indent=2, default=str)
        st.download_button(
            label="📥 Download Results JSON",
            data=results_json,
            file_name=f"egent_results_{timestamp}.json",
            mime="application/json"
        )
        
        # Clean up
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Egent: LLM-Powered Equivalent Width Measurement | 
    <a href="https://github.com/tingyuansen/Egent">GitHub</a> | 
    Ting et al. (2025)</p>
</div>
""", unsafe_allow_html=True)

