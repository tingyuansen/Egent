#!/usr/bin/env python3
"""
Batch EW Analysis: Run on All Magellan Spectra
===============================================

Runs the Egent EW analysis pipeline on all Magellan spectra.
Designed for overnight batch processing.

Features:
- Automatically discovers all Magellan spectra
- Matches Gaia IDs to pair IDs from catalog
- Saves results with gaia_id in filename
- Logs progress and errors
- Can resume from where it left off

Usage:
    python run_all_spectra.py                    # Run all spectra
    python run_all_spectra.py --mini             # Use mini model
    python run_all_spectra.py --workers 10       # Custom worker count
    python run_all_spectra.py --resume           # Skip already processed
    python run_all_spectra.py --dry-run          # List spectra without running
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import get_config
from run_ew import run_ew_analysis


def get_pair_mapping():
    """
    Create mapping from Gaia ID to (pair_id, component).

    Returns:
        dict: {gaia_id: (pair_id, component)}
    """
    mapping = {}

    cal_file = Path(__file__).parent / 'data' / 'spectra_good_calibration.csv'
    if cal_file.exists():
        df_cal = pd.read_csv(cal_file)
        for _, row in df_cal.iterrows():
            try:
                gaia_id = int(row['gaia_id'])
                pair_id = int(row['pair_id'])
                component = row['component']
                mapping[gaia_id] = (pair_id, component)
            except (ValueError, TypeError, KeyError):
                continue

    return mapping


def discover_spectra(use_good_only=True):
    """
    Find Magellan spectra to process.

    Args:
        use_good_only: If True, filter to well-calibrated spectra only

    Returns:
        list: List of (gaia_id, file_path) tuples
    """
    spectra_dir = Path(__file__).parent / 'spectra'

    good_list_file = Path(__file__).parent / 'data' / 'spectra_good_calibration.csv'
    good_gaia_ids = set()

    if use_good_only and good_list_file.exists():
        df_good = pd.read_csv(good_list_file)
        good_gaia_ids = set(df_good['gaia_id'].astype(int).tolist())
        print(f"Using {len(good_gaia_ids)} well-calibrated spectra from good list")

    spectra = []

    for f in spectra_dir.glob('*_magellan.csv'):
        name = f.stem.replace('_magellan', '')
        try:
            gaia_id = int(name)
            if use_good_only and good_gaia_ids and gaia_id not in good_gaia_ids:
                continue
            spectra.append((gaia_id, str(f)))
        except ValueError:
            continue

    return sorted(spectra)


def get_processed_gaia_ids(results_dir: Path) -> set:
    """Get set of Gaia IDs that have already been processed."""
    processed = set()
    for f in results_dir.glob('results_gaia*.json'):
        name = f.stem
        if name.startswith('results_gaia'):
            parts = name.replace('results_gaia', '').split('_')
            try:
                gaia_id = int(parts[0])
                processed.add(gaia_id)
            except ValueError:
                continue
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Run EW analysis on all Magellan spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_spectra.py                    # Run all spectra
    python run_all_spectra.py --mini             # Use mini model
    python run_all_spectra.py --workers 10       # Custom worker count
    python run_all_spectra.py --resume           # Skip already processed
    python run_all_spectra.py --dry-run          # List spectra without running
        """
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: auto based on model)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already processed spectra")
    parser.add_argument("--dry-run", action="store_true",
                        help="List spectra without processing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of spectra to process")
    parser.add_argument("--n-lines", type=int, default=None,
                        help="Limit lines per spectrum (for testing)")
    parser.add_argument("--all-spectra", action="store_true",
                        help="Process all spectra, not just well-calibrated ones")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--mini", action="store_true",
                        help="Use mini model (faster, cheaper)")

    args = parser.parse_args()

    config = get_config()

    # Discover spectra
    use_good_only = not args.all_spectra
    spectra = discover_spectra(use_good_only=use_good_only)
    print(f"Found {len(spectra)} Magellan spectra to process")

    # Get pair mapping
    pair_mapping = get_pair_mapping()
    print(f"Loaded pair mapping for {len(pair_mapping)} Gaia IDs")

    # Filter to those with pair info
    spectra_with_pairs = [(g, f, pair_mapping.get(g)) for g, f in spectra if g in pair_mapping]
    print(f"  {len(spectra_with_pairs)} spectra have pair info in catalog")

    # Determine output directory
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = config.get_output_dir(args.mini)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Resume: skip already processed
    if args.resume:
        processed = get_processed_gaia_ids(results_dir)
        spectra_with_pairs = [(g, f, p) for g, f, p in spectra_with_pairs if g not in processed]
        print(f"  {len(spectra_with_pairs)} remaining after resume filter")

    # Apply limit
    if args.limit:
        spectra_with_pairs = spectra_with_pairs[:args.limit]
        print(f"  Limited to {len(spectra_with_pairs)} spectra")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN: Spectra to process ===")
        for gaia_id, filepath, pair_info in spectra_with_pairs:
            pair_id, component = pair_info
            print(f"  Gaia {gaia_id} -> Pair {pair_id}{component}")
        return

    # Workers
    if args.workers is None:
        n_workers = config.get_workers(args.mini)
    else:
        n_workers = args.workers

    # Process spectra
    print("\n" + "="*70)
    print("BATCH EW ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend: {config.backend.upper()} | Model: {config.get_model(args.mini)}")
    print(f"Workers: {n_workers}")
    print(f"Output: {results_dir}")
    print("="*70)

    log_file = results_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    successful = 0
    failed = 0

    for i, (gaia_id, filepath, pair_info) in enumerate(spectra_with_pairs):
        pair_id, component = pair_info

        print(f"\n[{i+1}/{len(spectra_with_pairs)}] Gaia {gaia_id} (Pair {pair_id}{component})")
        print("-" * 50)

        start = time.time()

        try:
            results = run_ew_analysis(
                gaia_id=gaia_id,
                n_workers=n_workers,
                light_only=False,
                n_lines=args.n_lines,
                output_dir=str(results_dir),
                use_mini=args.mini,
            )

            elapsed = time.time() - start
            n_good = sum(1 for r in results if r.get('success') and not r.get('flagged'))
            n_flagged = sum(1 for r in results if r.get('flagged'))

            log_msg = f"SUCCESS: Gaia {gaia_id} | {n_good}/{len(results)} good, {n_flagged} flagged | {elapsed:.1f}s"
            print(f"  ✓ {log_msg}")
            successful += 1

        except Exception as e:
            elapsed = time.time() - start
            log_msg = f"FAILED: Gaia {gaia_id} | {str(e)[:50]} | {elapsed:.1f}s"
            print(f"  ✗ {log_msg}")
            failed += 1

        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {log_msg}\n")

    # Summary
    print("\n" + "="*70)
    print("BATCH COMPLETE")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Log: {log_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
