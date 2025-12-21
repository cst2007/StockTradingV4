# Stock Options Data Processor

A Python application for processing and merging stock options data from CSV files. This tool combines options side-by-side data with volatility Greeks data to create unified datasets for analysis.

## Overview

This processor automates the workflow of:
- Discovering matching pairs of options data files (side-by-side and volatility Greeks)
- Validating file formats and data integrity
- Merging related data by strike price
- Generating consolidated output files
- Archiving processed files to prevent duplicate processing

## Features

- **Automated File Discovery**: Intelligently matches options files with their corresponding Greeks files based on ticker, expiry date, and run date
- **Robust Validation**: Validates file headers, column counts, and data formats before processing
- **Data Cleaning**: Handles commas, percentages, and other formatting issues in numeric data
- **Duplicate Handling**: Automatically selects the most recent file when duplicates are detected
- **Comprehensive Logging**: Detailed logging for monitoring processing status and debugging
- **File Archiving**: Moves successfully processed files to a `processed` subdirectory
- **Multiple Output Formats**: Creates both a unified master file and per-symbol/date files

## Requirements

- Python 3.10 or higher
- pandas
- flask (for web UI)
- numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StockTradingV4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## File Naming Conventions

The processor expects files to follow specific naming patterns:

### Options Side-by-Side Files
```
<TICKER>-options-exp-<EXPIRY>-<TYPE>-side-by-side-<RUN_DATE>[-<SUFFIX>].csv
```
Example: `SPXL-options-exp-2025-12-05-test-side-by-side-11-28-2025.csv`

### Volatility Greeks Files
```
<TICKER>-volatility-greeks-exp-<EXPIRY>-<TYPE>-<RUN_DATE>[-<SUFFIX>].csv
```
Example: `SPXL-volatility-greeks-exp-2025-12-05-test-11-28-2025.csv`

**Date Formats:**
- `EXPIRY`: YYYY-MM-DD format
- `RUN_DATE`: MM-DD-YYYY format

## Input File Structure

### Side-by-Side File Headers
Expected 19 columns in this order:
```
Type, Last, Bid, Ask, Change, Volume, Open Int, IV, Last Trade, Strike,
Type, Last, Bid, Ask, Change, Volume, Open Int, IV, Last Trade
```
The first 9 columns represent call options, `Strike` is in the middle, and the last 9 columns represent put options.

### Volatility Greeks File Headers
Expected 17 columns in this order:
```
Last, Theor., IV, Delta, Gamma, Theta, Vega, Last Trade, Strike,
Last, Theor., IV, Delta, Gamma, Theta, Vega, Last Trade
```
The first 8 columns represent call Greeks, `Strike` is in the middle, and the last 8 columns represent put Greeks.

## Directory Structure

```
StockTradingV4/
├── app.py                  # Web UI Flask application
├── csv_processor.py        # Main processing script
├── generate_base_calculations.py  # Calculate exposures and rankings
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── WEB_UI_GUIDE.md        # Web UI documentation
├── templates/             # HTML templates
│   └── index.html        # Main web UI template
├── input/                 # Place CSV files here
│   └── processed/        # Archived processed files (auto-created)
└── output/
    ├── processing/       # Generated unified options files (auto-created)
    └── base_calculations/  # Generated base calculations (auto-created)
```

## Usage

### Option 1: Web UI (Recommended)

1. Place your CSV files in the `input/` directory

2. Start the web UI:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

4. For each pair:
   - Enter the spot value for the underlying asset
   - Click "Process Pair" to process that specific pair
   - View the results and output file names

5. Check the output files in:
   - `output/processing/` - Unified options data
   - `output/base_calculations/` - Calculated exposures and rankings

6. Successfully processed files are automatically moved to `input/processed/`

For detailed web UI instructions, see [WEB_UI_GUIDE.md](WEB_UI_GUIDE.md).

### Option 2: Command Line

1. Place your CSV files in the `input/` directory

2. Run the processor:
```bash
python csv_processor.py
```

3. Run base calculations:
```bash
python generate_base_calculations.py
```

4. Check the output in `output/processing/`:
   - `options_unified_raw.csv` - Master file containing all processed data
   - `options_unified_<SYMBOL>_<DATE>.csv` - Per-symbol/date files

5. Successfully processed files are moved to `input/processed/`

**Note:** When using the command line approach, the Spot value will be set to NA and needs to be filled in manually before running base calculations.

## Output Format

The unified output files contain the following columns:

| Column | Description |
|--------|-------------|
| Symbol | Stock ticker symbol (e.g., $SPX, SPXL) |
| Date | Run date in YYYY-MM-DD format |
| Expiry | Option expiration date |
| Spot | Spot price (placeholder, set to NA) |
| Strike | Strike price |
| call_delta | Call option delta |
| call_gamma | Call option gamma |
| call_theta | Call option theta |
| call_open_interest | Call open interest |
| call_volume | Call trading volume |
| Call_IV | Call implied volatility (merged from both sources) |
| Call_Vanna | Call vanna (placeholder, set to NA) |
| puts_delta | Put option delta |
| put_gamma | Put option gamma |
| put_theta | Put option theta |
| puts_open_interest | Put open interest |
| put_volume | Put trading volume |
| Put_IV | Put implied volatility (merged from both sources) |
| Put_Vanna | Put vanna (placeholder, set to NA) |

## How It Works

1. **Discovery Phase**: Scans the `input/` directory for CSV files matching the expected naming patterns
2. **Pairing**: Groups files by ticker, expiry date, and run date to find matching pairs
3. **Validation**: Checks file headers and column counts
4. **Loading**: Reads and parses CSV data, cleaning numeric values and percentages
5. **Merging**: Combines side-by-side and Greeks data on matching strike prices
6. **Output**: Generates unified CSV files sorted by symbol, date, expiry, and strike
7. **Archiving**: Moves processed files to the `processed/` subdirectory

## Special Handling

### Ticker Normalization
- `SPX` variations are normalized to `$SPX`
- All tickers are converted to uppercase

### Duplicate Files
When multiple files match the same ticker/expiry/run_date pattern:
- The most recently modified file is used
- Warnings are logged for transparency

### Missing Data
- Rows with invalid strike prices are dropped
- Invalid numeric values are logged but converted to NaN
- Files with no matching strikes between pairs are skipped

### Implied Volatility Merging
The processor combines IV data from both sources:
- Greeks file IV is preferred
- Side-by-side IV is used as fallback when Greeks IV is missing

## Logging

The application provides detailed logging output:
- **INFO**: Processing progress and summary statistics
- **WARNING**: Non-critical issues (duplicates, missing data, incomplete pairs)
- **ERROR**: Critical failures (file read errors, validation failures)
- **DEBUG**: Detailed diagnostic information

Log format:
```
[TIMESTAMP] [LEVEL] [csv_processor] [function_name] - message
```

## Error Handling

The processor is designed to be resilient:
- Invalid files are skipped with appropriate warnings
- Processing continues even if individual pairs fail
- Detailed error messages help diagnose issues
- File archiving errors don't halt the entire process

## Troubleshooting

### No output files generated
- Check that CSV files are in the `input/` directory
- Verify file names match the expected patterns
- Review logs for validation errors

### "Incomplete file set" warnings
- Ensure both side-by-side and Greeks files exist for each ticker/expiry/run_date combination
- Check file naming matches the expected patterns exactly

### "No matching strikes" warnings
- Verify that both files contain the same strike prices
- Check for data formatting issues in the Strike column

### Permission errors during archiving
- Ensure write permissions for the `input/processed/` directory
- Check that files aren't open in other applications

## Code Structure

### Data Classes
- `FileSetKey`: Unique identifier for file pairs (ticker, expiry, run_date)
- `FilePair`: Container for matched side-by-side and Greeks file paths
- `DiscoveryEntry`: Temporary storage during file discovery
- `ProcessingResult`: Combined dataframe and metadata

### Key Functions
- `discover_pairs()`: Finds and matches file pairs
- `validate_pair()`: Validates file headers and structure
- `load_side_df()`: Loads and parses side-by-side data
- `load_greeks_df()`: Loads and parses Greeks data
- `merge_pair()`: Combines data from both files
- `write_outputs()`: Generates output CSV files
- `archive_files()`: Moves processed files to archive

## Contributing

When contributing to this project:
1. Maintain the existing code style and structure
2. Add appropriate logging for new features
3. Update this README for any user-facing changes
4. Test with sample data before submitting

## License

[Specify your license here]

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify your input files match the expected format
3. Review the troubleshooting section above
4. Open an issue in the project repository
