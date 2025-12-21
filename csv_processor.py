"""
Stock Options Data Processor

This module processes and merges stock options CSV data from two sources:
1. Options side-by-side files (calls and puts with volume/OI/IV)
2. Volatility Greeks files (calls and puts with delta/gamma/theta)

The processor discovers matching file pairs, validates their structure,
merges them on strike price, and outputs unified CSV files for analysis.
"""

import csv
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
PROCESSED_DIR = INPUT_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "processing"

OPTIONS_PATTERN = re.compile(
    r"(?P<ticker>.+)-options-exp-(?P<expiry>\d{4}-\d{2}-\d{2})-.*-side-by-side-(?P<run_date>\d{2}-\d{2}-\d{4})(?:-\d+)?\.csv$",
    re.IGNORECASE,
)
GREEKS_PATTERN = re.compile(
    r"(?P<ticker>.+)-volatility-greeks-exp-(?P<expiry>\d{4}-\d{2}-\d{2})-.*-(?P<run_date>\d{2}-\d{2}-\d{4})(?:-\d+)?\.csv$",
    re.IGNORECASE,
)

EXPECTED_SIDE_HEADERS = [
    "Type",
    "Last",
    "Bid",
    "Ask",
    "Change",
    "Volume",
    "Open Int",
    "IV",
    "Last Trade",
    "Strike",
    "Type",
    "Last",
    "Bid",
    "Ask",
    "Change",
    "Volume",
    "Open Int",
    "IV",
    "Last Trade",
]

EXPECTED_GREEKS_HEADERS = [
    "Last",
    "Theor.",
    "IV",
    "Delta",
    "Gamma",
    "Theta",
    "Vega",
    "Last Trade",
    "Strike",
    "Last",
    "Theor.",
    "IV",
    "Delta",
    "Gamma",
    "Theta",
    "Vega",
    "Last Trade",
]

NAMES_SIDE = [
    "call_type",
    "call_last",
    "call_bid",
    "call_ask",
    "call_change",
    "call_volume",
    "call_open_interest",
    "call_iv_raw",
    "call_last_trade",
    "Strike",
    "put_type",
    "put_last",
    "put_bid",
    "put_ask",
    "put_change",
    "put_volume",
    "put_open_interest",
    "put_iv_raw",
    "put_last_trade",
]

NAMES_GREEKS = [
    "call_last",
    "call_theor",
    "call_iv",
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_vega",
    "call_last_trade",
    "Strike",
    "put_last",
    "put_theor",
    "put_iv",
    "puts_delta",
    "put_gamma",
    "put_theta",
    "put_vega",
    "put_last_trade",
]

OUTPUT_COLUMNS = [
    "Symbol",
    "Date",
    "Expiry",
    "Spot",
    "Strike",
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_open_interest",
    "call_volume",
    "Call_IV",
    "Call_Vanna",
    "puts_delta",
    "put_gamma",
    "put_theta",
    "puts_open_interest",
    "put_volume",
    "Put_IV",
    "Put_Vanna",
]

LOGGER = logging.getLogger("csv_processor")


@dataclass(frozen=True)
class FileSetKey:
    """Unique identifier for a set of related options data files.

    Attributes:
        ticker: Stock symbol (e.g., '$SPX', 'SPXL')
        expiry: Option expiration date in YYYY-MM-DD format
        run_date: Data collection date in MM-DD-YYYY format
    """
    ticker: str
    expiry: str
    run_date: str


@dataclass
class FilePair:
    """Container for matched side-by-side and Greeks file paths.

    Attributes:
        key: Unique identifier for this file pair
        side_path: Path to the options side-by-side CSV file
        greeks_path: Path to the volatility Greeks CSV file
    """
    key: FileSetKey
    side_path: Path
    greeks_path: Path


@dataclass
class DiscoveryEntry:
    """Temporary storage during file discovery phase.

    Attributes:
        side_path: Path to side-by-side file (if found)
        greeks_path: Path to Greeks file (if found)
    """
    side_path: Path | None = None
    greeks_path: Path | None = None


@dataclass
class ProcessingResult:
    """Result of successfully processing and merging a file pair.

    Attributes:
        dataframe: Merged and cleaned data
        pair: The file pair that was processed
    """
    dataframe: pd.DataFrame
    pair: FilePair


def normalize_ticker(raw_ticker: str) -> str:
    """Normalize ticker symbol to standard format.

    Converts ticker to uppercase and applies special handling for SPX
    to ensure consistency across different file naming conventions.

    Args:
        raw_ticker: Raw ticker string from filename

    Returns:
        Normalized ticker symbol (e.g., '$SPX', 'SPXL')
    """
    cleaned = raw_ticker.strip()
    if cleaned.upper().replace("$", "") == "SPX":
        return "$SPX"
    return cleaned.upper()


def parse_run_date(raw_date: str) -> str:
    """Convert run date from MM-DD-YYYY to YYYY-MM-DD format.

    Args:
        raw_date: Date string in MM-DD-YYYY format

    Returns:
        Date string in YYYY-MM-DD format
    """
    parsed = datetime.strptime(raw_date, "%m-%d-%Y")
    return parsed.strftime("%Y-%m-%d")


def read_header(file_path: Path) -> list[str]:
    """Read the header row from a CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        List of header column names
    """
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def normalize_headers(headers: list[str]) -> list[str]:
    """Normalize header strings for case-insensitive comparison.

    Args:
        headers: List of header strings

    Returns:
        List of normalized headers (stripped and lowercased)
    """
    return [header.strip().lower() for header in headers]


def validate_headers(
    file_path: Path,
    expected_headers: list[str],
    expected_columns: int,
    strike_index: int,
) -> bool:
    """Validate CSV file has expected structure and headers.

    Args:
        file_path: Path to CSV file to validate
        expected_headers: List of expected header names
        expected_columns: Expected number of columns
        strike_index: Index where 'Strike' column should be located

    Returns:
        True if file is valid, False otherwise
    """
    try:
        headers = read_header(file_path)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", file_path.name, exc)
        return False

    if len(headers) != expected_columns:
        LOGGER.error(
            "%s has %s columns, expected %s",
            file_path.name,
            len(headers),
            expected_columns,
        )
        return False

    normalized_actual = normalize_headers(headers)
    normalized_expected = normalize_headers(expected_headers)
    if normalized_actual != normalized_expected:
        LOGGER.error("%s has invalid headers", file_path.name)
        return False

    if strike_index >= len(headers) or headers[strike_index].strip().lower() != "strike":
        LOGGER.error("%s missing Strike column", file_path.name)
        return False

    return True


def parse_numeric_series(
    series: pd.Series,
    file_path: Path,
    field_name: str,
    drop_invalid: bool = False,
) -> pd.Series:
    """Parse and clean numeric data from CSV column.

    Removes commas from numbers and converts to numeric type.
    Logs warnings for invalid values that couldn't be converted.

    Args:
        series: Pandas series to parse
        file_path: Path to source file (for logging)
        field_name: Name of field (for logging)
        drop_invalid: Whether to drop invalid rows (not currently used)

    Returns:
        Cleaned numeric series with NaN for invalid values
    """
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    invalid_mask = numeric.isna() & cleaned.ne("")
    if invalid_mask.any():
        row_indices = numeric.index[invalid_mask].tolist()
        LOGGER.warning(
            "Invalid %s in %s at rows %s",
            field_name,
            file_path.name,
            row_indices,
        )
    if drop_invalid:
        return numeric
    return numeric


def parse_percent_series(series: pd.Series, file_path: Path, field_name: str) -> pd.Series:
    """Parse percentage values and convert to decimal.

    Removes '%' symbols and converts to decimal representation
    (e.g., '50%' becomes 0.50). Logs warnings for invalid values.

    Args:
        series: Pandas series containing percentage strings
        file_path: Path to source file (for logging)
        field_name: Name of field (for logging)

    Returns:
        Numeric series with decimal values (NaN for invalid values)
    """
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    invalid_mask = numeric.isna() & cleaned.ne("")
    if invalid_mask.any():
        row_indices = numeric.index[invalid_mask].tolist()
        LOGGER.warning(
            "Invalid %s in %s at rows %s",
            field_name,
            file_path.name,
            row_indices,
        )
    return numeric / 100


def discover_pairs(input_dir: Path) -> list[FilePair]:
    """Discover and match options file pairs in input directory.

    Scans for CSV files matching the expected naming patterns and
    groups them by ticker, expiry, and run date. Handles duplicates
    by selecting the most recently modified file.

    Args:
        input_dir: Directory containing CSV files

    Returns:
        List of complete file pairs ready for processing
    """
    entries: dict[FileSetKey, DiscoveryEntry] = {}
    total_files = 0

    for path in input_dir.glob("*.csv"):
        total_files += 1
        match = OPTIONS_PATTERN.match(path.name)
        if match:
            key = FileSetKey(
                ticker=normalize_ticker(match.group("ticker")),
                expiry=match.group("expiry"),
                run_date=match.group("run_date"),
            )
            entry = entries.setdefault(key, DiscoveryEntry())
            if entry.side_path is not None:
                entry.side_path = choose_newest(entry.side_path, path, key)
            else:
                entry.side_path = path
            continue

        match = GREEKS_PATTERN.match(path.name)
        if match:
            key = FileSetKey(
                ticker=normalize_ticker(match.group("ticker")),
                expiry=match.group("expiry"),
                run_date=match.group("run_date"),
            )
            entry = entries.setdefault(key, DiscoveryEntry())
            if entry.greeks_path is not None:
                entry.greeks_path = choose_newest(entry.greeks_path, path, key)
            else:
                entry.greeks_path = path
            continue

        LOGGER.debug("Skipping unrecognized file %s", path.name)

    LOGGER.info("Discovered %s csv files", total_files)
    pairs: list[FilePair] = []
    for key, entry in entries.items():
        if entry.side_path is None or entry.greeks_path is None:
            LOGGER.warning(
                "Incomplete file set for %s/%s/%s",
                key.ticker,
                key.expiry,
                key.run_date,
            )
            continue
        pairs.append(FilePair(key=key, side_path=entry.side_path, greeks_path=entry.greeks_path))

    return pairs


def choose_newest(current_path: Path, new_path: Path, key: FileSetKey) -> Path:
    """Select the most recently modified file when duplicates exist.

    Args:
        current_path: Currently selected file path
        new_path: Newly discovered file path
        key: File set identifier (for logging)

    Returns:
        Path to the most recently modified file
    """
    current_mtime = current_path.stat().st_mtime
    new_mtime = new_path.stat().st_mtime
    if new_mtime > current_mtime:
        LOGGER.warning(
            "Duplicate file for %s/%s/%s: using %s",
            key.ticker,
            key.expiry,
            key.run_date,
            new_path.name,
        )
        return new_path
    LOGGER.warning(
        "Duplicate file for %s/%s/%s: keeping %s",
        key.ticker,
        key.expiry,
        key.run_date,
        current_path.name,
    )
    return current_path


def validate_pair(pair: FilePair) -> bool:
    """Validate both files in a pair have correct structure.

    Args:
        pair: File pair to validate

    Returns:
        True if both files are valid, False otherwise
    """
    side_ok = validate_headers(pair.side_path, EXPECTED_SIDE_HEADERS, 19, 9)
    greeks_ok = validate_headers(pair.greeks_path, EXPECTED_GREEKS_HEADERS, 17, 8)
    return side_ok and greeks_ok


def load_side_df(side_path: Path) -> pd.DataFrame:
    """Load and parse options side-by-side CSV file.

    Reads call and put option data including volume, open interest,
    and implied volatility. Cleans numeric values and percentages.

    Args:
        side_path: Path to side-by-side CSV file

    Returns:
        DataFrame with parsed side-by-side data
    """
    data = pd.read_csv(side_path, header=0, names=NAMES_SIDE, dtype=str)
    data["Strike"] = parse_numeric_series(data["Strike"], side_path, "Strike", drop_invalid=True)
    data["call_volume"] = parse_numeric_series(data["call_volume"], side_path, "call_volume")
    data["call_open_interest"] = parse_numeric_series(
        data["call_open_interest"], side_path, "call_open_interest"
    )
    data["put_volume"] = parse_numeric_series(data["put_volume"], side_path, "put_volume")
    data["put_open_interest"] = parse_numeric_series(
        data["put_open_interest"], side_path, "put_open_interest"
    )
    data["call_iv_raw"] = parse_percent_series(data["call_iv_raw"], side_path, "call_iv_raw")
    data["put_iv_raw"] = parse_percent_series(data["put_iv_raw"], side_path, "put_iv_raw")
    data = data.dropna(subset=["Strike"])
    return data[
        [
            "Strike",
            "call_volume",
            "call_open_interest",
            "call_iv_raw",
            "put_volume",
            "put_open_interest",
            "put_iv_raw",
        ]
    ]


def load_greeks_df(greeks_path: Path) -> pd.DataFrame:
    """Load and parse volatility Greeks CSV file.

    Reads call and put Greeks data including delta, gamma, theta,
    and implied volatility. Cleans numeric values and percentages.

    Args:
        greeks_path: Path to Greeks CSV file

    Returns:
        DataFrame with parsed Greeks data
    """
    data = pd.read_csv(greeks_path, header=0, names=NAMES_GREEKS, dtype=str)
    data["Strike"] = parse_numeric_series(data["Strike"], greeks_path, "Strike", drop_invalid=True)
    data["call_iv"] = parse_percent_series(data["call_iv"], greeks_path, "call_iv")
    data["put_iv"] = parse_percent_series(data["put_iv"], greeks_path, "put_iv")
    data["call_delta"] = parse_numeric_series(data["call_delta"], greeks_path, "call_delta")
    data["call_gamma"] = parse_numeric_series(data["call_gamma"], greeks_path, "call_gamma")
    data["call_theta"] = parse_numeric_series(data["call_theta"], greeks_path, "call_theta")
    data["puts_delta"] = parse_numeric_series(data["puts_delta"], greeks_path, "puts_delta")
    data["put_gamma"] = parse_numeric_series(data["put_gamma"], greeks_path, "put_gamma")
    data["put_theta"] = parse_numeric_series(data["put_theta"], greeks_path, "put_theta")
    data = data.dropna(subset=["Strike"])
    return data[
        [
            "Strike",
            "call_delta",
            "call_gamma",
            "call_theta",
            "puts_delta",
            "put_gamma",
            "put_theta",
            "call_iv",
            "put_iv",
        ]
    ]


def merge_pair(pair: FilePair) -> ProcessingResult | None:
    """Merge side-by-side and Greeks data for a file pair.

    Loads both files, merges them on strike price, combines IV data
    from both sources, and adds metadata columns.

    Args:
        pair: File pair to merge

    Returns:
        ProcessingResult if successful, None if merge fails
    """
    try:
        side_df = load_side_df(pair.side_path)
        greeks_df = load_greeks_df(pair.greeks_path)
    except Exception as exc:
        LOGGER.error("Failed to read %s/%s: %s", pair.side_path.name, pair.greeks_path.name, exc)
        return None

    merged = pd.merge(greeks_df, side_df, on="Strike", how="inner")
    if merged.empty:
        LOGGER.warning(
            "No matching strikes between files for %s/%s/%s",
            pair.key.ticker,
            pair.key.expiry,
            pair.key.run_date,
        )
        return None

    merged["Call_IV"] = merged["call_iv"].combine_first(merged["call_iv_raw"])
    merged["Put_IV"] = merged["put_iv"].combine_first(merged["put_iv_raw"])
    merged["Symbol"] = pair.key.ticker
    merged["Date"] = parse_run_date(pair.key.run_date)
    merged["Expiry"] = pair.key.expiry
    merged["Spot"] = pd.NA
    merged["Call_Vanna"] = pd.NA
    merged["Put_Vanna"] = pd.NA
    merged["puts_open_interest"] = merged["put_open_interest"]

    merged = merged.drop(
        columns=[
            "call_iv",
            "put_iv",
            "call_iv_raw",
            "put_iv_raw",
            "put_open_interest",
        ]
    )

    merged = merged.dropna(subset=["Strike"])
    merged = merged[OUTPUT_COLUMNS]
    return ProcessingResult(dataframe=merged, pair=pair)


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def unique_destination(destination: Path) -> Path:
    """Generate unique file path by adding timestamp if file exists.

    Args:
        destination: Desired file path

    Returns:
        Unique file path (original or with timestamp suffix)
    """
    if not destination.exists():
        return destination
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return destination.with_name(f"{destination.stem}-{timestamp}{destination.suffix}")


def archive_files(pair: FilePair) -> None:
    """Move processed files to archive directory.

    Moves both side-by-side and Greeks files to the processed
    subdirectory to prevent reprocessing.

    Args:
        pair: File pair to archive
    """
    ensure_directory(PROCESSED_DIR)
    for path in [pair.side_path, pair.greeks_path]:
        destination = unique_destination(PROCESSED_DIR / path.name)
        try:
            shutil.move(str(path), str(destination))
            LOGGER.info("Archived %s to %s", path.name, destination.name)
        except PermissionError as exc:
            LOGGER.error("Cannot move %s to processed: %s", path.name, exc)
        except Exception as exc:
            LOGGER.error("Failed to archive %s: %s", path.name, exc)


def write_outputs(results: list[ProcessingResult]) -> list[Path]:
    """Generate output CSV files from processing results.

    Creates a master unified file containing all data, plus individual
    files split by symbol and date.

    Args:
        results: List of successfully processed results

    Returns:
        List of output file paths created
    """
    if not results:
        LOGGER.info("No valid pairs to process")
        return []

    ensure_directory(OUTPUT_DIR)
    combined = pd.concat([result.dataframe for result in results], ignore_index=True)
    combined = combined.sort_values(by=["Symbol", "Date", "Expiry", "Strike"])
    master_path = OUTPUT_DIR / "options_unified_raw.csv"
    combined.to_csv(master_path, index=False)

    output_paths = [master_path]
    for (symbol, date), group in combined.groupby(["Symbol", "Date"], dropna=False):
        output_path = OUTPUT_DIR / f"options_unified_{symbol}_{date}.csv"
        group.to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths


def process_pairs() -> None:
    """Main processing workflow.

    Discovers file pairs, validates them, merges data, writes outputs,
    and archives successfully processed files.
    """
    ensure_directory(INPUT_DIR)
    pairs = discover_pairs(INPUT_DIR)
    LOGGER.info("Processing %s complete pairs", len(pairs))

    results: list[ProcessingResult] = []
    for pair in pairs:
        if not validate_pair(pair):
            continue
        result = merge_pair(pair)
        if result is None:
            continue
        results.append(result)

    output_paths = write_outputs(results)

    if not output_paths:
        return

    for result in results:
        archive_files(result.pair)

    LOGGER.info("Processed %s pairs", len(results))


def configure_logging() -> None:
    """Configure logging format and level for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s",
    )


def main() -> None:
    """Application entry point."""
    configure_logging()
    process_pairs()


if __name__ == "__main__":
    main()
