"""
Generate Base Calculations from Options Unified Data

This script processes unified options data and calculates various exposure metrics
using proper dealer positioning methodology:

- DEX (Delta Exposure) - Dealer-signed, no spot scaling
- GEX (Gamma Exposure) - Moneyness-based sign functions
- GEX_SKEW - Put/Call gamma skew
- VOL_SHOCK - Vega exposure (volatility shock impact)
- THETA_EXPO - Time decay exposure
- Z-Scores - Cross-strike statistical significance
- Window Totals - Aggregated metrics per expiry

One output file is generated per Symbol/Date pair.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output" / "processing"
OUTPUT_DIR = BASE_DIR / "output" / "base_calculations"

LOGGER = logging.getLogger("base_calculations")

# Contract multiplier
M = 100

# Epsilon for near-ATM threshold (strikes within this % of spot are neutral)
EPSILON_PERCENT = 0.001  # 0.1% threshold


def calculate_moneyness_signs(spot: pd.Series, strike: pd.Series, epsilon_pct: float = EPSILON_PERCENT) -> tuple:
    """Calculate moneyness sign functions for GEX.

    For calls: +1 if S > K (OTM calls, dealers short gamma)
               -1 if S < K (ITM calls, dealers long gamma)
                0 if |S - K| < ε (ATM, neutral)

    For puts:  +1 if S < K (OTM puts, dealers short gamma)
               -1 if S > K (ITM puts, dealers long gamma)
                0 if |S - K| < ε (ATM, neutral)

    Args:
        spot: Spot price series
        strike: Strike price series
        epsilon_pct: Threshold percentage for ATM region

    Returns:
        Tuple of (sign_call, sign_put) as numpy arrays
    """
    # Calculate absolute and relative distance
    distance = np.abs(spot - strike)
    epsilon = spot * epsilon_pct

    # Initialize signs
    sign_call = np.zeros(len(spot))
    sign_put = np.zeros(len(spot))

    # Where spot > strike
    otm_call_mask = spot > strike
    # Where spot < strike
    otm_put_mask = spot < strike
    # Where near ATM
    atm_mask = distance < epsilon

    # Assign signs (dealers are SHORT gamma for OTM options they sold)
    sign_call = np.where(otm_call_mask & ~atm_mask, 1,
                np.where(~otm_call_mask & ~atm_mask, -1, 0))

    sign_put = np.where(otm_put_mask & ~atm_mask, 1,
               np.where(~otm_put_mask & ~atm_mask, -1, 0))

    return sign_call, sign_put


def calculate_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all exposure metrics from unified options data.

    Implements proper dealer positioning methodology:
    - DEX: Dealer-signed delta exposure (no spot scaling)
    - GEX: Moneyness-signed gamma exposure
    - GEX_SKEW: Put/call gamma asymmetry
    - VOL_SHOCK: Vega exposure for volatility sensitivity
    - THETA_EXPO: Time decay exposure

    Args:
        df: DataFrame containing unified options data with:
            Spot, Strike, call_delta, put_delta, call_gamma, put_gamma,
            call_theta, put_theta, call_vega, put_vega,
            call_open_interest, puts_open_interest

    Returns:
        DataFrame with calculated exposure metrics
    """
    result = df.copy()

    # Extract base data
    spot = result['Spot']
    strike = result['Strike']

    # Fill NaN values with 0 for Greeks and OI
    call_delta = result['call_delta'].fillna(0)
    call_gamma = result['call_gamma'].fillna(0)
    call_theta = result['call_theta'].fillna(0)
    call_vega = result['call_vega'].fillna(0)
    call_oi = result['call_open_interest'].fillna(0)

    put_delta = result['puts_delta'].fillna(0)
    put_gamma = result['put_gamma'].fillna(0)
    put_theta = result['put_theta'].fillna(0)
    put_vega = result['put_vega'].fillna(0)
    put_oi = result['puts_open_interest'].fillna(0)

    # ========================================================================
    # D) DEX Formulas (Dealer-signed, no spot scaling)
    # ========================================================================
    # Dealers are SHORT options, so they have OPPOSITE delta exposure
    # Negative sign represents dealer hedging flow

    result['CALL_DEX'] = -call_delta * call_oi * M
    result['PUT_DEX'] = -put_delta * put_oi * M
    result['DEX'] = result['CALL_DEX'] + result['PUT_DEX']

    # ========================================================================
    # E) GEX Formulas (Moneyness-based signs)
    # ========================================================================
    # Calculate moneyness signs
    sign_call, sign_put = calculate_moneyness_signs(spot, strike)

    result['CALL_GEX'] = sign_call * call_gamma * call_oi * M
    result['PUT_GEX'] = sign_put * put_gamma * put_oi * M
    result['GEX'] = result['CALL_GEX'] + result['PUT_GEX']

    # GEX Skew (per-strike and will be used for totals)
    result['GEX_SKEW'] = result['PUT_GEX'] - result['CALL_GEX']

    # ========================================================================
    # F) VOL_SHOCK (Vega Exposure) - replaces VEX
    # ========================================================================
    # Dealer short vega exposure (negative sign)
    result['CALL_VEGA_EXPO'] = -call_vega * call_oi * M
    result['PUT_VEGA_EXPO'] = -put_vega * put_oi * M
    result['VOL_SHOCK'] = result['CALL_VEGA_EXPO'] + result['PUT_VEGA_EXPO']

    # ========================================================================
    # G) THETA_EXPO (Time Decay Exposure)
    # ========================================================================
    # Dealer short theta exposure (they collect premium decay)
    result['CALL_THETA_EXPO'] = -call_theta * call_oi * M
    result['PUT_THETA_EXPO'] = -put_theta * put_oi * M
    result['THETA_EXPO'] = result['CALL_THETA_EXPO'] + result['PUT_THETA_EXPO']

    # ========================================================================
    # Additional Metrics
    # ========================================================================
    # IV columns (already in unified data)
    call_iv = result['Call_IV'].fillna(0)
    put_iv = result['Put_IV'].fillna(0)

    # IV x OI (useful for premium-weighted positioning)
    result['Call_IVxOI'] = call_iv * call_oi
    result['Put_IVxOI'] = put_iv * put_oi
    result['IVxOI'] = result['Call_IVxOI'] + result['Put_IVxOI']

    # Position Metrics
    result['Distance_to_Spot'] = np.abs(strike - spot)
    epsilon = 1e-9
    result['Rel_Dist'] = np.where(
        spot.notna(),
        result['Distance_to_Spot'] / (spot + epsilon),
        np.nan
    )
    result['OI_Imbalance'] = call_oi - put_oi

    return result


def calculate_window_totals(df: pd.DataFrame) -> dict:
    """Calculate window-wide totals for all exposure metrics.

    Args:
        df: DataFrame with per-strike exposures

    Returns:
        Dictionary with total metrics for the expiry window
    """
    totals = {}

    # DEX Totals
    totals['DEX_total'] = df['DEX'].sum()
    totals['CALL_DEX_total'] = df['CALL_DEX'].sum()
    totals['PUT_DEX_total'] = df['PUT_DEX'].sum()

    # DEX in dollar terms (if spot available)
    if 'Spot' in df.columns and df['Spot'].notna().any():
        spot_value = df['Spot'].dropna().iloc[0]
        totals['DEX_$_total'] = totals['DEX_total'] * spot_value
    else:
        totals['DEX_$_total'] = np.nan

    # GEX Totals
    totals['CALL_GEX_total'] = df['CALL_GEX'].sum()
    totals['PUT_GEX_total'] = df['PUT_GEX'].sum()
    totals['NET_GEX_total'] = totals['CALL_GEX_total'] + totals['PUT_GEX_total']
    totals['GEX_SKEW_total'] = totals['PUT_GEX_total'] - totals['CALL_GEX_total']

    # VOL_SHOCK Totals
    totals['VEGA_EXPO_total'] = df['VOL_SHOCK'].sum()
    totals['CALL_VEGA_EXPO_total'] = df['CALL_VEGA_EXPO'].sum()
    totals['PUT_VEGA_EXPO_total'] = df['PUT_VEGA_EXPO'].sum()

    # THETA Totals
    totals['THETA_EXPO_total'] = df['THETA_EXPO'].sum()
    totals['CALL_THETA_EXPO_total'] = df['CALL_THETA_EXPO'].sum()
    totals['PUT_THETA_EXPO_total'] = df['PUT_THETA_EXPO'].sum()

    # Additional totals
    totals['IVxOI_total'] = df['IVxOI'].sum()
    totals['Total_Call_OI'] = df['call_open_interest'].sum()
    totals['Total_Put_OI'] = df['puts_open_interest'].sum()
    totals['Total_OI'] = totals['Total_Call_OI'] + totals['Total_Put_OI']

    return totals


def calculate_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add z-score columns for cross-strike statistical significance.

    Z-scores show how many standard deviations each strike's metric is
    from the mean across all strikes in the window.

    Args:
        df: DataFrame with calculated exposure metrics

    Returns:
        DataFrame with added z-score columns
    """
    result = df.copy()

    # List of metrics to calculate z-scores for
    metrics = [
        'DEX',
        'GEX',
        'GEX_SKEW',
        'VOL_SHOCK',
        'THETA_EXPO',
        'IVxOI',
    ]

    # Also add z-scores for IV if available
    if 'Call_IV' in result.columns:
        # Use average of call and put IV for the z-score
        result['Avg_IV'] = (result['Call_IV'].fillna(0) + result['Put_IV'].fillna(0)) / 2
        metrics.append('Avg_IV')

    # Calculate z-scores for each metric
    for metric in metrics:
        if metric in result.columns:
            values = result[metric]
            mean_val = values.mean()
            std_val = values.std()

            # Avoid division by zero
            if std_val > 0:
                result[f'{metric}_z'] = (values - mean_val) / std_val
            else:
                result[f'{metric}_z'] = 0.0

    return result


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking columns based on absolute values of key metrics.

    Rankings are in descending order (1 = highest absolute value).

    Args:
        df: DataFrame with calculated exposure metrics

    Returns:
        DataFrame with added ranking columns
    """
    result = df.copy()

    # Metrics to rank
    rank_metrics = [
        'DEX',
        'GEX',
        'GEX_SKEW',
        'VOL_SHOCK',
        'THETA_EXPO',
        'IVxOI',
        'OI_Imbalance',
    ]

    # Rank by absolute values, descending (1 = highest absolute value)
    for metric in rank_metrics:
        if metric in result.columns:
            result[f'{metric}_Rank'] = result[metric].abs().rank(method='min', ascending=False)

    return result


def normalize_decimals(df: pd.DataFrame, decimal_places: int = 3) -> pd.DataFrame:
    """Normalize numeric columns to a fixed number of decimal places.

    Args:
        df: DataFrame to normalize
        decimal_places: Number of decimal places to round to (default: 3)

    Returns:
        DataFrame with normalized decimal values
    """
    numeric_columns = df.select_dtypes(include=["float64", "float32"]).columns
    for col in numeric_columns:
        df[col] = df[col].round(decimal_places)
    return df


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def process_unified_files() -> list[Path]:
    """Process all unified options files and generate base calculations.

    Returns:
        List of output file paths created
    """
    # Find all options_unified files (excluding the raw master file)
    unified_files = [
        f for f in INPUT_DIR.glob("options_unified_*.csv")
        if f.name != "options_unified_raw.csv"
    ]

    if not unified_files:
        LOGGER.warning("No unified options files found in %s", INPUT_DIR)
        return []

    LOGGER.info("Found %s unified file(s) to process", len(unified_files))

    ensure_directory(OUTPUT_DIR)
    output_paths = []

    for file_path in unified_files:
        LOGGER.info("Processing %s", file_path.name)

        try:
            # Read unified options data
            df = pd.read_csv(file_path)

            # Validate required columns
            required_cols = ['Spot', 'Strike', 'call_delta', 'puts_delta',
                           'call_gamma', 'put_gamma', 'call_theta', 'put_theta',
                           'call_vega', 'put_vega', 'call_open_interest', 'puts_open_interest']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                LOGGER.error("Missing required columns in %s: %s", file_path.name, missing_cols)
                continue

            # Calculate exposures
            result_df = calculate_exposures(df)

            # Calculate window totals
            totals = calculate_window_totals(result_df)
            LOGGER.info("  Window Totals:")
            LOGGER.info("    DEX_total: %.2f", totals['DEX_total'])
            LOGGER.info("    NET_GEX_total: %.2f", totals['NET_GEX_total'])
            LOGGER.info("    GEX_SKEW_total: %.2f", totals['GEX_SKEW_total'])
            LOGGER.info("    VEGA_EXPO_total: %.2f", totals['VEGA_EXPO_total'])
            LOGGER.info("    THETA_EXPO_total: %.2f", totals['THETA_EXPO_total'])

            # Add z-scores
            result_df = calculate_z_scores(result_df)

            # Add rankings
            result_df = add_rankings(result_df)

            # Normalize decimals
            result_df = normalize_decimals(result_df, decimal_places=3)

            # Add totals as metadata columns (same value for all rows)
            for key, value in totals.items():
                result_df[key] = value

            # Generate output filename: base_calculations_SYMBOL_DATE.csv
            # Extract symbol and date from input filename
            # Input format: options_unified_SYMBOL_DATE.csv
            parts = file_path.stem.replace("options_unified_", "").split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                date = parts[1]
                output_filename = f"base_calculations_{symbol}_{date}.csv"
            else:
                output_filename = f"base_calculations_{file_path.stem}.csv"

            output_path = OUTPUT_DIR / output_filename
            result_df.to_csv(output_path, index=False)
            output_paths.append(output_path)

            LOGGER.info("  ✓ Generated %s", output_filename)

        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", file_path.name, exc)
            import traceback
            LOGGER.error(traceback.format_exc())
            continue

    return output_paths


def configure_logging() -> None:
    """Configure logging format and level for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    )


def main() -> None:
    """Application entry point."""
    configure_logging()

    LOGGER.info("Starting base calculations generation...")
    LOGGER.info("Using improved dealer positioning methodology:")
    LOGGER.info("  - DEX: Dealer-signed delta exposure")
    LOGGER.info("  - GEX: Moneyness-based gamma exposure")
    LOGGER.info("  - GEX_SKEW: Put/call gamma asymmetry")
    LOGGER.info("  - VOL_SHOCK: Vega-based volatility sensitivity")
    LOGGER.info("  - THETA_EXPO: Time decay exposure")
    LOGGER.info("  - Z-scores: Cross-strike statistical significance")
    LOGGER.info("")

    output_paths = process_unified_files()

    if output_paths:
        LOGGER.info("Successfully generated %s base calculation file(s)", len(output_paths))
        for path in output_paths:
            LOGGER.info("  - %s", path.name)
    else:
        LOGGER.info("No files were generated")


if __name__ == "__main__":
    main()
