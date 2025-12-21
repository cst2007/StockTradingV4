# Stock Options Processing Web UI Guide

## Overview

The web UI provides a visual interface for processing stock options data pairs. It allows you to:
- View all available options file pairs in the input folder
- Enter spot values for each pair
- Process individual pairs with specified spot values
- Generate both unified options output and base calculations

## Starting the Web UI

1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Using the Web UI

### Main Interface

The main page displays all available option pairs found in the `input/` folder. Each pair is shown in a card with:

- **Ticker Symbol**: The stock/index symbol (e.g., $SPX, SPXL, TSLL)
- **Expiry Date**: The option expiration date
- **Run Date**: The data collection date
- **File Information**: Names of the side-by-side and Greeks CSV files

### Processing a Pair

For each pair:

1. **Enter Spot Value**: Input the current spot price of the underlying asset
   - Must be a positive number
   - Can include decimal values (e.g., 150.25)

2. **Click "Process Pair"**: Initiates processing with the following steps:
   - Validates the file pair
   - Merges options side-by-side and Greeks data
   - Sets the spot value
   - Calculates all exposure metrics (GEX, DEX, VEX, Theta, etc.)
   - Adds rankings based on absolute values
   - Generates output files

3. **View Results**: After successful processing, you'll see:
   - Success message
   - Output filename in `output/processing/`
   - Base calculations filename in `output/base_calculations/`

### Output Files

Two output files are generated per processed pair:

1. **Unified Options File**: `output/processing/options_unified_{TICKER}_{DATE}.csv`
   - Contains merged options data with all calculated exposures

2. **Base Calculations File**: `output/base_calculations/base_calculations_{TICKER}_{DATE}.csv`
   - Contains the same data with additional ranking columns

### File Processing

After successful processing:
- The original input files are automatically moved to `input/processed/`
- This prevents reprocessing the same files
- The pair will no longer appear in the web UI on refresh

## Example Workflow

1. Place your CSV files in the `input/` folder:
   - `SPXL-options-exp-2025-12-05-test-side-by-side-11-28-2025.csv`
   - `SPXL-volatility-greeks-exp-2025-12-05-test-11-28-2025.csv`

2. Start the web UI: `python app.py`

3. Open http://localhost:5000 in your browser

4. Find the SPXL pair card

5. Enter the spot value (e.g., 150.25)

6. Click "Process Pair"

7. Check the success message and output files

8. View results in `output/processing/` and `output/base_calculations/`

## File Naming Convention

The system expects specific file naming patterns:

**Options Side-by-Side Files:**
```
{TICKER}-options-exp-{YYYY-MM-DD}-*-side-by-side-{MM-DD-YYYY}.csv
```

**Volatility Greeks Files:**
```
{TICKER}-volatility-greeks-exp-{YYYY-MM-DD}-*-{MM-DD-YYYY}.csv
```

Files with the same ticker, expiry, and run date are automatically matched as pairs.

## Troubleshooting

### No Pairs Found
- Ensure CSV files are in the `input/` folder
- Check that files follow the expected naming convention
- Verify that both side-by-side and Greeks files exist for each ticker/date combination

### Processing Errors
- Verify that the spot value is a valid positive number
- Check that CSV files have the expected column structure
- Review the console/terminal output for detailed error messages

### Connection Issues
- Ensure port 5000 is not already in use
- Try accessing http://127.0.0.1:5000 instead of localhost
- Check firewall settings if accessing from a different machine

## Technical Details

- **Backend**: Flask (Python)
- **Data Processing**: pandas, numpy
- **Port**: 5000 (default)
- **Host**: 0.0.0.0 (accessible from network)

## Notes

- The web UI processes one pair at a time
- Each pair requires manual spot value entry
- Files are automatically archived after successful processing
- The interface uses AJAX for smooth processing without page reloads
- All logging is visible in the console/terminal running the Flask app
