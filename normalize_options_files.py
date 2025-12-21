"""
Normalize decimal values in existing options_unified CSV files.

This script rounds all numeric columns to 3 decimal places to fix
floating-point precision issues.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "processing"


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


def normalize_file(file_path: Path) -> None:
    """Normalize decimals in a single CSV file.

    Args:
        file_path: Path to CSV file to normalize
    """
    print(f"Normalizing {file_path.name}...")
    df = pd.read_csv(file_path)
    df = normalize_decimals(df, decimal_places=3)
    df.to_csv(file_path, index=False)
    print(f"  ✓ Normalized {file_path.name}")


def main() -> None:
    """Main entry point."""
    # Find all options_unified CSV files
    csv_files = list(OUTPUT_DIR.glob("options_unified*.csv"))

    if not csv_files:
        print("No options_unified files found to normalize.")
        return

    print(f"Found {len(csv_files)} file(s) to normalize:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    print()
    for csv_file in csv_files:
        normalize_file(csv_file)

    print(f"\nSuccessfully normalized {len(csv_files)} file(s).")


if __name__ == "__main__":
    main()
