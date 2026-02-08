# CLAUDE.md

Guide for AI assistants working on the StockTradingV4 codebase.

## Project Overview

Stock options data processor that automates broker-exported CSV file workflows: discover, validate, merge, and analyze options data. Calculates dealer positioning exposures (DEX, GEX, VOL_SHOCK, THETA) and generates trading signals for Cash-Secured Put (CSP) and Covered Call (CC) strategies.

**Core concept:** Market makers (dealers) are SHORT options. All exposure formulas apply a negative sign to represent dealer hedging flow. Z-scores measure statistical significance of each strike relative to other strikes in the same expiry window.

## Tech Stack

- **Language:** Python 3.10+ (uses modern type hints like `Path | None`)
- **Web framework:** Flask with Jinja2 templates
- **Data processing:** pandas, NumPy
- **Database:** None — all data is file-based (CSV in, CSV out)
- **No CI/CD, no automated tests, no linter config**

## Project Structure

```
StockTradingV4/
├── app.py                         # Flask web app — orchestrates full pipeline
├── csv_processor.py               # File discovery, validation, and merge logic
├── generate_base_calculations.py  # Exposure calculations, z-scores, decision tables
├── weekly_strike_scores.py        # Standalone CLI scoring tool
├── diagnose_pair.py               # Debugging utility for file pairs
├── normalize_options_files.py     # Decimal normalizer utility
├── requirements.txt               # pandas, flask, numpy
│
├── input/                         # Drop broker CSV files here
│   └── processed/                 # Archived files after processing
├── output/
│   ├── processing/                # Stage 1: merged unified options CSVs
│   └── base_calculations/         # Stage 2: exposures + z-scores + decisions
│
├── templates/
│   ├── index.html                 # Pair selection & processing UI
│   └── results.html               # Decision table analysis & visualization
│
└── Documentation/
    ├── README.md
    ├── DOCUMENTATION.md            # Full technical reference (~850 lines)
    ├── WEB_UI_GUIDE.md
    ├── COLUMN_MAPPING.md
    ├── COMBINED_VIEW_GUIDE.md
    └── EQUITY_RESULTS_PAGE.md
```

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `csv_processor.py` | ~933 | Regex-based file discovery, header validation, pair merge via inner join on Strike |
| `generate_base_calculations.py` | ~961 | All exposure formulas (DEX/GEX/VOL_SHOCK/THETA), z-scores, decision tables, equity interpretations |
| `app.py` | ~260 | Flask routes, pipeline orchestration, JSON API |
| `weekly_strike_scores.py` | ~611 | Standalone CLI for weekly strike scoring |

## Running the Application

### Web UI (primary workflow)
```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:5000
```

### CLI
```bash
python csv_processor.py                  # Merge pairs from input/
python generate_base_calculations.py     # Calculate exposures on merged files
python weekly_strike_scores.py --input output/processing/options_unified_<SYMBOL>_<DATE>.csv
```

### Diagnostic
```bash
python diagnose_pair.py <TICKER> <EXPIRY> <RUN_DATE>
```

## Processing Pipeline

The full pipeline runs via `POST /process_pair` in `app.py`:

```
CSV Input → discover_pairs() → validate_pair() → merge_pair()
  → inject spot value → calculate_exposures() → calculate_window_totals()
  → calculate_z_scores() → add_decision_tables()
  → add_equity_interpretations() → add_rankings()
  → normalize_decimals() → write output CSVs
```

**Two output files per run:**
- `output/processing/options_unified_<SYMBOL>_<DATE>.csv` (21 columns)
- `output/base_calculations/base_calculations_<SYMBOL>_<DATE>.csv` (80+ columns)

## Input File Conventions

Files must follow these naming patterns (regex-matched):
```
OPTIONS:  <ticker>-options-exp-<YYYY-MM-DD>-*-side-by-side-<MM-DD-YYYY>.csv
GREEKS:   <ticker>-volatility-greeks-exp-<YYYY-MM-DD>-*-<MM-DD-YYYY>.csv
```

- Side-by-side files: 19 columns (9 call + Strike + 9 put)
- Greeks files: 17 columns (8 call + Strike + 8 put)
- Encoding: UTF-8 with BOM support (UTF-8-SIG)
- Numeric values may contain commas (`1,234.56`) or percent signs (`27.95%`)

## Flask API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Discover file pairs, render index.html |
| `/process_pair` | POST | Full pipeline for one pair (JSON request/response) |
| `/results` | GET | List base_calculations files, render results.html |
| `/api/calculation/<filename>` | GET | Return calculation data as JSON |

## Key Constants and Formulas

**Contract multiplier:** `M = 100`
**ATM epsilon:** `0.1%` (strikes within this % of spot are neutral)

### DEX (Delta Exposure)
```
CALL_DEX = -delta_c * OI_c * M
PUT_DEX  = -delta_p * OI_p * M
DEX = CALL_DEX + PUT_DEX
```

### GEX (Gamma Exposure)
```
sign = +1 (OTM, dealers short gamma), -1 (ITM), 0 (ATM within epsilon)
CALL_GEX = sign_call * gamma_c * OI_c * M
PUT_GEX  = sign_put  * gamma_p * OI_p * M
GEX = CALL_GEX + PUT_GEX
```

### Other exposures
- **GEX_SKEW** = PUT_GEX - CALL_GEX
- **VOL_SHOCK** = -(vega_c * OI_c + vega_p * OI_p) * M
- **THETA_EXPO** = -(theta_c * OI_c + theta_p * OI_p) * M

### Z-Score Bands
Z-scores classify strikes into 6 bands: `<= -2.0`, `-2.0 to -1.5`, `-1.5 to -0.75`, `-0.75 to +0.75`, `+0.75 to +1.5`, `>= +1.5`

### Equity Combined Signals
Combined signals use emoji indicators:
- `🟢` Strong Support (Buyable Dip)
- `🔴` Support Likely to Fail
- `🧲` Strong Resistance (Fade Zone)
- `🚀` Breakout / Trend Zone
- `🟡` Weak / Conditional Support
- `⚪` Neutral Zone

## Coding Conventions

- **Style:** PEP 8 (4-space indentation), no formal linter enforced
- **Type hints:** Python 3.10+ syntax (`Path | None`, not `Optional[Path]`)
- **Data structures:** `@dataclass` for DTOs (`FileSetKey`, `FilePair`, `ProcessingResult`)
- **Logging:** Python `logging` module, not `print()`
- **File paths:** `pathlib.Path` throughout, never raw strings
- **Variable naming:** `snake_case` for functions/variables, `CamelCase` for classes, `ALL_CAPS` for constants
- **Docstrings:** All functions have docstrings with `Args:` and `Returns:` sections
- **Column naming:** snake_case internally (`call_delta`), Title_Case in outputs (`Call_IV`)
- **Numeric handling:** Commas stripped, percentages converted (e.g., `27.95%` -> `0.2795`), NaN for invalid values

## Ticker Normalization

- SPX variations (`spx`, `SPX`, etc.) are normalized to `$SPX`
- All other tickers are uppercased

## Date Formats

- **In filenames (run_date):** `MM-DD-YYYY`
- **In filenames (expiry):** `YYYY-MM-DD`
- **In output CSVs:** `YYYY-MM-DD`

## Important Patterns

- **Duplicate file handling:** When duplicates exist, the most recently modified file is used (with a warning)
- **IV merge strategy:** Greeks-file IV is preferred; side-by-side IV is used as fallback
- **Column parsing:** Header validation by name (not position), regex cleaning of pandas suffixes (`.1`, `.2`), case-insensitive matching
- **Archiving is disabled:** `archive_files()` call is commented out in `app.py:151` to allow reprocessing

## Common Modification Scenarios

### Adding a new exposure metric
1. Add calculation function in `generate_base_calculations.py`
2. Wire it into the pipeline in `app.py` (between `calculate_exposures` and `normalize_decimals`)
3. Add z-score calculation in `calculate_z_scores()`
4. Add decision table mapping in `add_decision_tables()`
5. Include new columns in the API response in `app.py:get_calculation_data()`
6. Update `results.html` to display the new columns

### Adding a new input file format
1. Add regex pattern in `csv_processor.py` (in `discover_pairs()`)
2. Add header validation logic in `validate_pair()`
3. Add loader function (like `load_side_df()` / `load_greeks_df()`)
4. Update `merge_pair()` to incorporate new data

### Adding a new API endpoint
1. Add route in `app.py`
2. Follow existing patterns: security checks on filenames, try/except with LOGGER.error, return `jsonify()`

## Testing

No automated test suite exists. Manual testing approach:
1. Place sample CSV files in `input/`
2. Process via web UI at `localhost:5000`
3. Verify output files in `output/processing/` and `output/base_calculations/`
4. Use `diagnose_pair.py` for debugging file pair issues
5. Check results page at `localhost:5000/results`

## Dependencies

Only three runtime dependencies (all in `requirements.txt`):
- `pandas` — DataFrame processing and CSV I/O
- `flask` — Web server and routing
- `numpy` — Numerical computations

No dev dependencies, no lock file, no virtual environment configuration checked in.
