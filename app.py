"""
Flask Web Application for Stock Options Processing

This application provides a web UI to:
1. View available options file pairs in the input folder
2. Enter spot values for each pair
3. Process individual pairs with the specified spot value
"""

import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd

from csv_processor import (
    discover_pairs,
    validate_pair,
    merge_pair,
    archive_files,
    ensure_directory,
    normalize_decimals,
    INPUT_DIR,
    OUTPUT_DIR,
    FilePair,
    FileSetKey,
)
from generate_base_calculations import (
    calculate_exposures,
    calculate_z_scores,
    add_decision_tables,
    add_equity_interpretations,
    add_rankings,
    calculate_window_totals,
    OUTPUT_DIR as BASE_CALC_DIR,
)

app = Flask(__name__)
LOGGER = logging.getLogger("options_web_ui")


def configure_logging():
    """Configure logging format and level for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    )


@app.route("/")
def index():
    """Render the main page with available pairs."""
    pairs = discover_pairs(INPUT_DIR)

    # Convert pairs to a format suitable for the template
    pairs_data = []
    for pair in pairs:
        pairs_data.append({
            "ticker": pair.key.ticker,
            "expiry": pair.key.expiry,
            "run_date": pair.key.run_date,
            "side_file": pair.side_path.name,
            "greeks_file": pair.greeks_path.name,
            "pair_id": f"{pair.key.ticker}_{pair.key.expiry}_{pair.key.run_date}"
        })

    return render_template("index.html", pairs=pairs_data)


@app.route("/process_pair", methods=["POST"])
def process_pair():
    """Process a specific pair with the provided spot value."""
    try:
        data = request.json
        ticker = data.get("ticker")
        expiry = data.get("expiry")
        run_date = data.get("run_date")
        spot_value = data.get("spot_value")

        if not all([ticker, expiry, run_date, spot_value]):
            return jsonify({"success": False, "error": "Missing required parameters"}), 400

        try:
            spot_value = float(spot_value)
        except ValueError:
            return jsonify({"success": False, "error": "Invalid spot value"}), 400

        # Find the matching pair
        pairs = discover_pairs(INPUT_DIR)
        target_key = FileSetKey(ticker=ticker, expiry=expiry, run_date=run_date)
        target_pair = None

        for pair in pairs:
            if pair.key == target_key:
                target_pair = pair
                break

        if not target_pair:
            return jsonify({"success": False, "error": "Pair not found"}), 404

        # Validate the pair
        valid, validation_error = validate_pair(target_pair)
        if not valid:
            return jsonify({"success": False, "error": f"Invalid file pair: {validation_error}"}), 400

        # Merge the pair
        result = merge_pair(target_pair)
        if result is None:
            return jsonify({"success": False, "error": "Failed to merge files"}), 500

        # Set the spot value
        result.dataframe["Spot"] = spot_value

        # Calculate exposures
        result.dataframe = calculate_exposures(result.dataframe)

        # Calculate window totals
        totals = calculate_window_totals(result.dataframe)

        # Add z-scores
        result.dataframe = calculate_z_scores(result.dataframe)

        # Add decision tables
        result.dataframe = add_decision_tables(result.dataframe)

        # Add equity interpretations
        result.dataframe = add_equity_interpretations(result.dataframe)

        # Add rankings
        result.dataframe = add_rankings(result.dataframe)

        # Add totals as metadata columns
        for key, value in totals.items():
            result.dataframe[key] = value

        # Normalize decimals
        result.dataframe = normalize_decimals(result.dataframe, decimal_places=3)

        # Save the output
        ensure_directory(OUTPUT_DIR)
        output_filename = f"options_unified_{ticker}_{result.dataframe['Date'].iloc[0]}.csv"
        output_path = OUTPUT_DIR / output_filename
        result.dataframe.to_csv(output_path, index=False)

        # Also save base calculations
        from generate_base_calculations import OUTPUT_DIR as BASE_CALC_OUTPUT_DIR, ensure_directory as ensure_base_dir
        ensure_base_dir(BASE_CALC_OUTPUT_DIR)
        base_calc_filename = f"base_calculations_{ticker}_{result.dataframe['Date'].iloc[0]}.csv"
        base_calc_path = BASE_CALC_OUTPUT_DIR / base_calc_filename
        result.dataframe.to_csv(base_calc_path, index=False)

        # Archiving disabled to allow reprocessing of pairs with updated data
        # archive_files(target_pair)

        LOGGER.info("Successfully processed %s/%s/%s with spot=%s", ticker, expiry, run_date, spot_value)

        return jsonify({
            "success": True,
            "message": f"Successfully processed {ticker}",
            "output_file": output_filename,
            "base_calc_file": base_calc_filename
        })

    except Exception as exc:
        LOGGER.error("Error processing pair: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/results")
def results():
    """Render the results page showing decision table analysis."""
    # Get list of available base calculation files
    base_calc_files = []
    if BASE_CALC_DIR.exists():
        for file in BASE_CALC_DIR.glob("base_calculations_*.csv"):
            base_calc_files.append({
                "filename": file.name,
                "name": file.stem.replace("base_calculations_", "")
            })

    return render_template("results.html", files=base_calc_files)


@app.route("/api/calculation/<filename>")
def get_calculation_data(filename):
    """Get calculation data for a specific file."""
    try:
        # Security: only allow base_calculations files
        if not filename.startswith("base_calculations_"):
            return jsonify({"success": False, "error": "Invalid filename"}), 400

        file_path = BASE_CALC_DIR / filename
        if not file_path.exists():
            return jsonify({"success": False, "error": "File not found"}), 404

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Get window totals from first row
        window_totals = {}
        if len(df) > 0:
            for col in df.columns:
                if col.endswith('_total') or col.startswith('Total_'):
                    window_totals[col] = float(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else None

        # Select key columns for display
        display_columns = [
            'Strike', 'Spot',
            'DEX', 'DEX_z', 'DEX_z_band',
            'GEX', 'GEX_z', 'GEX_z_band',
            'GEX_SKEW', 'GEX_SKEW_z', 'GEX_SKEW_z_band',
            'VOL_SHOCK', 'VOL_SHOCK_z', 'VOL_SHOCK_z_band',
            'DEX_CSP_Action', 'DEX_CC_Action',
            'GEX_CSP_Action', 'GEX_CC_Action',
            'GEX_SKEW_CSP_Action', 'GEX_SKEW_CC_Action',
            'VOL_SHOCK_CSP_Action', 'VOL_SHOCK_CC_Action',
            'GEX_SKEW_Equity_Meaning',
            'CSP_Hard_Blocks', 'CC_Hard_Blocks',
            'CSP_Combined_Signal', 'CC_Combined_Signal',
            'call_open_interest', 'puts_open_interest', 'OI_Imbalance',
            'Call_IV', 'Put_IV',
        ]

        # Filter to only columns that exist
        available_columns = [col for col in display_columns if col in df.columns]
        result_df = df[available_columns].copy()

        # Round numeric columns for cleaner display
        for col in result_df.select_dtypes(include=['float64']).columns:
            result_df[col] = result_df[col].round(3)

        # Convert to dict for JSON
        data = result_df.to_dict('records')

        # Add summary statistics
        summary = {
            'total_strikes': len(df),
            'csp_strong_count': len(df[df['CSP_Combined_Signal'].str.contains('STRONG', na=False)]),
            'cc_strong_count': len(df[df['CC_Combined_Signal'].str.contains('STRONG', na=False)]),
            'csp_blocked_count': len(df[df['CSP_Hard_Blocks'].str.contains('CSP_NO', na=False)]),
            'cc_conditional_count': len(df[df['CC_Hard_Blocks'].str.contains('CC_CONDITIONAL', na=False)]),
        }

        return jsonify({
            "success": True,
            "data": data,
            "window_totals": window_totals,
            "summary": summary,
            "symbol": df['Symbol'].iloc[0] if 'Symbol' in df.columns and len(df) > 0 else "Unknown",
            "date": df['Date'].iloc[0] if 'Date' in df.columns and len(df) > 0 else "Unknown",
            "expiry": df['Expiry'].iloc[0] if 'Expiry' in df.columns and len(df) > 0 else "Unknown",
        })

    except Exception as exc:
        LOGGER.error("Error getting calculation data: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    configure_logging()
    app.run(debug=True, host="0.0.0.0", port=5000)
