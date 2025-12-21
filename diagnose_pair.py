#!/usr/bin/env python3
"""
Diagnostic script to check if a specific pair of files can be processed.
Helps identify issues with file pairing and validation.
"""

import sys
import logging
from pathlib import Path
from csv_processor import (
    discover_pairs,
    validate_pair,
    INPUT_DIR,
    FileSetKey,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def diagnose_files(ticker: str, expiry: str, run_date: str):
    """Diagnose why a specific pair might not be working."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING PAIR: {ticker} / {expiry} / {run_date}")
    print(f"{'='*60}\n")

    # Step 1: Check if input directory exists
    print(f"1. Checking input directory: {INPUT_DIR}")
    if not INPUT_DIR.exists():
        print(f"   ❌ ERROR: Input directory does not exist!")
        return
    print(f"   ✓ Input directory exists\n")

    # Step 2: List all CSV files in input directory
    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"2. Found {len(csv_files)} CSV files in input directory:")
    for f in csv_files:
        print(f"   - {f.name}")
    print()

    # Step 3: Discover all pairs
    print(f"3. Discovering pairs...")
    pairs = discover_pairs(INPUT_DIR)
    print(f"   ✓ Found {len(pairs)} valid pairs\n")

    # Step 4: List all discovered pairs
    print(f"4. Available pairs:")
    target_key = FileSetKey(ticker=ticker.upper(), expiry=expiry, run_date=run_date)
    found_target = False

    for pair in pairs:
        is_target = pair.key == target_key
        marker = ">>> TARGET <<<" if is_target else ""
        print(f"   - {pair.key.ticker} / {pair.key.expiry} / {pair.key.run_date} {marker}")
        if is_target:
            found_target = True
            print(f"     Options: {pair.side_path.name}")
            print(f"     Greeks:  {pair.greeks_path.name}")
    print()

    # Step 5: Check if target pair was found
    if not found_target:
        print(f"5. ❌ TARGET PAIR NOT FOUND!")
        print(f"\n   Looking for:")
        print(f"   - Ticker: {ticker.upper()}")
        print(f"   - Expiry: {expiry}")
        print(f"   - Run Date: {run_date}")
        print(f"\n   Expected filenames (in {INPUT_DIR}):")
        print(f"   - {ticker.lower()}-options-exp-{expiry}-*-side-by-side-{run_date}.csv")
        print(f"   - {ticker.lower()}-volatility-greeks-exp-{expiry}-*-{run_date}.csv")
        print(f"\n   Possible reasons:")
        print(f"   1. Files not uploaded to input directory")
        print(f"   2. Filenames don't match the expected pattern")
        print(f"   3. Files might be in the 'processed' subdirectory (already processed)")
        print(f"   4. Run date mismatch in filenames")
        return

    print(f"5. ✓ TARGET PAIR FOUND!\n")

    # Step 6: Validate the pair
    print(f"6. Validating pair...")
    target_pair = None
    for pair in pairs:
        if pair.key == target_key:
            target_pair = pair
            break

    if validate_pair(target_pair):
        print(f"   ✓ Pair validation PASSED!")
        print(f"\n{'='*60}")
        print(f"SUCCESS: Pair is ready to be processed!")
        print(f"{'='*60}\n")
    else:
        print(f"   ❌ Pair validation FAILED!")
        print(f"\n   This means the files have incorrect structure/headers.")
        print(f"   Check the log messages above for specific validation errors.")
        print(f"\n{'='*60}")
        print(f"FAILURE: Fix validation errors before processing")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diagnose_pair.py <TICKER> <EXPIRY> <RUN_DATE>")
        print("Example: python diagnose_pair.py TSLL 2025-12-26 12-20-2025")
        sys.exit(1)

    ticker = sys.argv[1]
    expiry = sys.argv[2]
    run_date = sys.argv[3]

    diagnose_files(ticker, expiry, run_date)
