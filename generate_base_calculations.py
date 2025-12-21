"""
Generate Base Calculations from Options Unified Data

This script processes unified options data and calculates various exposure metrics:
- GEX (Gamma Exposure)
- DEX (Delta Exposure)
- VEX (Vanna Exposure)
- Theta Exposure
- IV x OI
- Position Metrics
- Rankings (based on absolute values of key metrics)

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


def calculate_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all exposure metrics from unified options data.

    Args:
        df: DataFrame containing unified options data

    Returns:
        DataFrame with calculated exposure metrics
    """
    result = df.copy()

    # Keep Spot and Strike as-is (may contain NaN)
    spot = result['Spot']
    strike = result['Strike']

    # Fill NaN values with 0 for greeks and OI (missing data should be treated as 0)
    call_delta = result['call_delta'].fillna(0)
    call_gamma = result['call_gamma'].fillna(0)
    call_theta = result['call_theta'].fillna(0)
    call_oi = result['call_open_interest'].fillna(0)
    call_iv = result['Call_IV'].fillna(0)
    call_vanna = result['Call_Vanna'].fillna(0)

    puts_delta = result['puts_delta'].fillna(0)
    put_gamma = result['put_gamma'].fillna(0)
    put_theta = result['put_theta'].fillna(0)
    put_oi = result['puts_open_interest'].fillna(0)
    put_iv = result['Put_IV'].fillna(0)
    put_vanna = result['Put_Vanna'].fillna(0)

    # GEX Exposure (will be NaN if Spot is NaN)
    spot_squared = spot ** 2
    result['Call_GEX'] = call_gamma * spot_squared * call_oi * 100
    result['Put_GEX'] = put_gamma * spot_squared * put_oi * 100
    result['Net_GEX'] = result['Call_GEX'] - result['Put_GEX']
    result['Total_GEX'] = result['Call_GEX'] + result['Put_GEX']

    # DEX Exposure (will be NaN if Spot is NaN)
    result['Call_DEX'] = call_delta * spot * call_oi * 100
    result['Put_DEX'] = puts_delta * spot * put_oi * 100
    result['Net_DEX'] = result['Call_DEX'] - result['Put_DEX']

    # Vanna Exposure (doesn't depend on Spot)
    result['Call_VEX'] = call_vanna * call_oi * 100
    result['Put_VEX'] = put_vanna * put_oi * 100
    result['Net_VEX'] = result['Call_VEX'] - result['Put_VEX']

    # Theta Exposure (doesn't depend on Spot)
    result['Call_Theta_Exp'] = call_theta * call_oi * 100
    result['Put_Theta_Exp'] = put_theta * put_oi * 100
    result['Net_TEX'] = result['Call_Theta_Exp'] - result['Put_Theta_Exp']
    result['Net_Theta_Exp'] = result['Call_Theta_Exp'] + result['Put_Theta_Exp']

    # IV x OI (doesn't depend on Spot)
    result['Call_IVxOI'] = call_iv * call_oi
    result['Put_IVxOI'] = put_iv * put_oi
    result['IVxOI'] = result['Call_IVxOI'] + result['Put_IVxOI']

    # Position Metrics (will be NaN if Spot is NaN)
    result['Distance_to_Spot'] = np.abs(strike - spot)
    epsilon = 1e-9
    # Only calculate Rel_Dist where Spot is not NaN
    result['Rel_Dist'] = np.where(
        spot.notna(),
        result['Distance_to_Spot'] / (spot + epsilon),
        np.nan
    )
    result['OI_Imbalance'] = call_oi - put_oi

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

    # Rank by absolute values, descending (1 = highest absolute value)
    # method='min' gives the same rank to ties (e.g., 1, 2, 2, 4)
    # ascending=False means highest values get lowest rank numbers
    result['Net_GEX_Rank'] = result['Net_GEX'].abs().rank(method='min', ascending=False)
    result['Net_DEX_Rank'] = result['Net_DEX'].abs().rank(method='min', ascending=False)
    result['Net_VEX_Rank'] = result['Net_VEX'].abs().rank(method='min', ascending=False)
    result['Net_Theta_Rank'] = result['Net_Theta_Exp'].abs().rank(method='min', ascending=False)
    result['IVxOI_Rank'] = result['IVxOI'].abs().rank(method='min', ascending=False)
    result['OI_Imbalance_Rank'] = result['OI_Imbalance'].abs().rank(method='min', ascending=False)

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

            # Calculate exposures
            result_df = calculate_exposures(df)

            # Add rankings
            result_df = add_rankings(result_df)

            # Normalize decimals
            result_df = normalize_decimals(result_df, decimal_places=3)

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
    output_paths = process_unified_files()

    if output_paths:
        LOGGER.info("Successfully generated %s base calculation file(s)", len(output_paths))
        for path in output_paths:
            LOGGER.info("  - %s", path.name)
    else:
        LOGGER.info("No files were generated")


if __name__ == "__main__":
    main()
