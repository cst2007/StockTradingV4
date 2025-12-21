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
from generate_base_calculations import calculate_exposures, add_rankings

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
        if not validate_pair(target_pair):
            return jsonify({"success": False, "error": "Invalid file pair"}), 400

        # Merge the pair
        result = merge_pair(target_pair)
        if result is None:
            return jsonify({"success": False, "error": "Failed to merge files"}), 500

        # Set the spot value
        result.dataframe["Spot"] = spot_value

        # Calculate exposures
        result.dataframe = calculate_exposures(result.dataframe)

        # Add rankings
        result.dataframe = add_rankings(result.dataframe)

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

        # Archive the processed files
        archive_files(target_pair)

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


if __name__ == "__main__":
    configure_logging()
    app.run(debug=True, host="0.0.0.0", port=5000)
