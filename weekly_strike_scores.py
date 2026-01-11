"""
Weekly Options Strike Scoring

Computes per-strike DEX/GEX/GEX_SKEW/VOL_SHOCK exposures, z-scores, and
decision-table interpretations for a single expiry (or grouped by expiry).

VOL_SHOCK is a proxy based on vega × OI (not true vanna).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("weekly_strike_scores")
M = 100

COLUMN_ALIASES = {
    "strike": ["strike", "Strike"],
    "spot": ["spot", "Spot"],
    "expiry": ["expiry", "Expiry"],
    "call_oi": ["call_oi", "call_open_interest"],
    "put_oi": ["put_oi", "puts_open_interest", "put_open_interest"],
    "call_delta": ["call_delta"],
    "put_delta": ["put_delta", "puts_delta"],
    "call_gamma": ["call_gamma"],
    "put_gamma": ["put_gamma"],
    "call_vega": ["call_vega"],
    "put_vega": ["put_vega"],
}

REQUIRED_COLUMNS = [
    "strike",
    "spot",
    "call_oi",
    "put_oi",
    "call_delta",
    "put_delta",
    "call_gamma",
    "put_gamma",
    "call_vega",
    "put_vega",
]

BANDS = [
    (lambda z: z <= -2.0),
    (lambda z: -2.0 < z <= -1.5),
    (lambda z: -1.5 < z <= -0.75),
    (lambda z: -0.75 < z < 0.75),
    (lambda z: 0.75 <= z < 1.5),
    (lambda z: z >= 1.5),
]

DEX_TABLE = [
    (
        "Extreme downside acceleration; forced dealer selling",
        "Never CSP (not support)",
        "CC only if already long shares and want protection; expect downside",
        "Support likely to fail hard",
    ),
    (
        "Strong downside pressure",
        "Avoid CSP",
        "CC acceptable near call resistance; don’t chase",
        "Weak/temporary support",
    ),
    (
        "Mild downside bias; weak support",
        "CSP only with strong GEX_z + level confirmation",
        "CC becomes attractive (rallies fade faster)",
        "Support needs confirmation",
    ),
    (
        "Neutral directional pressure",
        "Ideal CSP zone",
        "Ideal CC zone",
        "Normal technical S/R",
    ),
    (
        "Upside capped; dealer absorption on rips",
        "Strong CSP",
        "CC only at clear resistance (upside may stall early)",
        "Strong support; resistance holds",
    ),
    (
        "Extreme upside suppression; forced selling into rallies",
        "Best CSP",
        "Avoid CC (pin / slow grind through strike risk)",
        "Strong resistance / pin risk",
    ),
]

GEX_TABLE = [
    (
        "Extreme negative gamma → fast moves, air pockets, “trap premium”",
        "Avoid CSP (unless intentionally playing crash-bounce + strict exits)",
        "CC only if you want downside continuation protection; avoid “naked timing”",
        "Levels break violently / trend expansion",
    ),
    (
        "Strong instability; supports break easier",
        "Avoid / very small size only with confirmation",
        "Prefer waiting; if already long shares, CC can be ok for protection",
        "S/R unreliable",
    ),
    (
        "Mild negative gamma; trend/vol expansion more likely",
        "CSP only with structural support + confirmation (HVL/put wall)",
        "CC becomes more attractive (moves can extend; rips fade less)",
        "Breakouts more likely",
    ),
    (
        "Neutral regime; mixed behavior",
        "“Normal” CSP selection (then use DEX_z + levels)",
        "“Normal” CC selection",
        "Normal TA applies",
    ),
    (
        "Positive gamma pocket; mean reversion; magnets stronger",
        "Preferred CSP zone (dips damped)",
        "CC less attractive for big premium unless near resistance/magnet",
        "Levels act as magnets / fades work",
    ),
    (
        "Strong positive gamma → tight ranges, pinning, “sticky” levels",
        "Best CSP (esp. at/under key supports)",
        "CC only at clear call resistance/upper band; premium may be small",
        "Very strong S/R, chop/pin",
    ),
]

GEX_SKEW_TABLE = [
    (
        "Strong call-side dominance",
        "Avoid CSP unless everything else is perfect",
        "Strongly favors CC at resistance",
        "Resistance very strong; support fragile",
    ),
    (
        "Call-side dominance",
        "CSP only if DEX_z >= 0 AND (NET_GEX_total > 0 or GEX_z positive) AND VOL_SHOCK_z not high",
        "CC favored",
        "Sell rips / fade strength",
    ),
    (
        "Mild call dominance",
        "CSP selective",
        "CC slightly favored",
        "Resistance > support",
    ),
    (
        "Balanced",
        "Neutral",
        "Neutral",
        "No side advantage",
    ),
    (
        "Mild put dominance",
        "CSP favored",
        "CC selective",
        "Support > resistance",
    ),
    (
        "Strong put-side dominance",
        "Strongly favors CSP",
        "Avoid CC unless at extreme resistance",
        "Strong support / dip-buy behavior",
    ),
]

VOL_SHOCK_TABLE = [
    (
        "Low vol sensitivity pocket",
        "CSP ok if DEX/GEX ok",
        "CC ok",
        "Slow moves; levels break cleanly",
    ),
    (
        "Low vol sensitivity",
        "CSP ok",
        "CC ok",
        "Breakouts often stall",
    ),
    (
        "Mild",
        "Normal",
        "Normal",
        "Controlled moves",
    ),
    (
        "Neutral",
        "Ideal zone",
        "Ideal zone",
        "Clean technical reactions",
    ),
    (
        "High vol sensitivity",
        "CSP only if strong support (GEX_z >= 0, DEX_z >= 0)",
        "CC only at strong resistance",
        "Whipsaws; fast moves; widen stops",
    ),
    (
        "IV shock leverage zone",
        "Avoid CSP (premium trap / gap risk)",
        "Avoid CC unless intentionally want vol expansion risk",
        "Explosive failure, gaps, slippage",
    ),
]


def configure_logging() -> None:
    """Configure logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    )


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected canonical names."""
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required columns exist."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert columns to numeric values."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input ranges and enforce numeric types."""
    require_columns(df, REQUIRED_COLUMNS)
    numeric_columns = REQUIRED_COLUMNS
    df = coerce_numeric(df, numeric_columns)

    if (df["call_oi"] < 0).any() or (df["put_oi"] < 0).any():
        raise ValueError("Open interest must be non-negative.")
    if (df["call_gamma"] < 0).any() or (df["put_gamma"] < 0).any():
        raise ValueError("Gamma must be non-negative.")

    return df


def delta_valid_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask of rows with valid delta ranges."""
    call_ok = df["call_delta"].between(0, 1, inclusive="both")
    put_ok = df["put_delta"].between(-1, 0, inclusive="both")
    return call_ok & put_ok


def zscore(series: pd.Series, eligible_mask: pd.Series) -> pd.Series:
    """Compute z-score using population std-dev (ddof=0)."""
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    eligible = series[eligible_mask]
    if eligible.empty:
        return result
    mean = eligible.mean()
    std = eligible.std(ddof=0)
    if std == 0 or np.isnan(std):
        result.loc[eligible_mask] = 0.0
        return result
    result.loc[eligible_mask] = (eligible - mean) / std
    return result


def decision_table(z_value: float, table: list[tuple[str, str, str, str]]) -> tuple[str, str, str, str]:
    """Resolve decision table entry for a given z-score."""
    if np.isnan(z_value):
        return ("", "", "", "")
    for band, entry in zip(BANDS, table):
        if band(z_value):
            return entry
    return table[-1]


def apply_decision_table(series: pd.Series, table: list[tuple[str, str, str, str]], prefix: str) -> pd.DataFrame:
    """Apply decision table to a z-score series."""
    meanings = []
    csp_actions = []
    cc_actions = []
    equity_actions = []
    for value in series.tolist():
        meaning, csp, cc, equity = decision_table(value, table)
        meanings.append(meaning)
        csp_actions.append(csp)
        cc_actions.append(cc)
        equity_actions.append(equity)
    return pd.DataFrame(
        {
            f"{prefix}_meaning": meanings,
            f"{prefix}_CSP_action": csp_actions,
            f"{prefix}_CC_action": cc_actions,
            f"{prefix}_Equity_action": equity_actions,
        },
        index=series.index,
    )


def compute_group_scores(df: pd.DataFrame, window_percent: float | None) -> pd.DataFrame:
    """Compute exposures and z-scores for a single expiry group."""
    result = df.copy()

    valid_delta = delta_valid_mask(result)

    result["CALL_DEX"] = -result["call_delta"] * result["call_oi"] * M
    result["PUT_DEX"] = -result["put_delta"] * result["put_oi"] * M
    result["DEX"] = result["CALL_DEX"] + result["PUT_DEX"]

    spot = result["spot"]
    strike = result["strike"]
    eps = np.maximum(0.0001 * spot, 0.01)
    distance = (spot - strike).abs()
    atm_mask = distance < eps
    invalid_spot = spot.isna() | strike.isna()
    sign_call = np.where(
        invalid_spot,
        np.nan,
        np.where(atm_mask, 0, np.where(spot > strike, 1, -1)),
    )
    sign_put = np.where(
        invalid_spot,
        np.nan,
        np.where(atm_mask, 0, np.where(spot < strike, 1, -1)),
    )

    result["CALL_GEX"] = sign_call * result["call_gamma"] * result["call_oi"] * M
    result["PUT_GEX"] = sign_put * result["put_gamma"] * result["put_oi"] * M
    result["GEX"] = result["CALL_GEX"] + result["PUT_GEX"]
    result["GEX_SKEW"] = result["PUT_GEX"] - result["CALL_GEX"]

    result["CALL_VEGA_EXPO"] = -result["call_vega"] * result["call_oi"] * M
    result["PUT_VEGA_EXPO"] = -result["put_vega"] * result["put_oi"] * M
    result["VEGA_EXPO"] = result["CALL_VEGA_EXPO"] + result["PUT_VEGA_EXPO"]
    result["VOL_SHOCK"] = result["VEGA_EXPO"]

    if window_percent is None:
        window_mask = pd.Series(True, index=result.index)
    else:
        window_mask = (strike - spot).abs() <= (spot * window_percent)

    dex_mask = valid_delta & result["DEX"].notna() & window_mask
    gex_mask = result["GEX"].notna() & window_mask
    gex_skew_mask = result["GEX_SKEW"].notna() & window_mask
    vol_shock_mask = result["VOL_SHOCK"].notna() & window_mask

    result["DEX_z"] = zscore(result["DEX"], dex_mask)
    result["GEX_z"] = zscore(result["GEX"], gex_mask)
    result["GEX_SKEW_z"] = zscore(result["GEX_SKEW"], gex_skew_mask)
    result["VOL_SHOCK_z"] = zscore(result["VOL_SHOCK"], vol_shock_mask)

    result = pd.concat(
        [
            result,
            apply_decision_table(result["DEX_z"], DEX_TABLE, "DEX"),
            apply_decision_table(result["GEX_z"], GEX_TABLE, "GEX"),
            apply_decision_table(result["GEX_SKEW_z"], GEX_SKEW_TABLE, "GEX_SKEW"),
            apply_decision_table(result["VOL_SHOCK_z"], VOL_SHOCK_TABLE, "VOL_SHOCK"),
        ],
        axis=1,
    )

    return result


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CSP/CC combined decision flags."""
    result = df.copy()

    csp_flag = []
    cc_flag = []

    for _, row in result.iterrows():
        dex_z = row.get("DEX_z")
        gex_z = row.get("GEX_z")
        vol_z = row.get("VOL_SHOCK_z")
        skew_z = row.get("GEX_SKEW_z")

        csp_state = "CSP_OK"
        if pd.isna(dex_z) or pd.isna(gex_z) or pd.isna(vol_z) or pd.isna(skew_z):
            csp_state = "CSP_CONDITIONAL"
        if dex_z <= -1.5 or gex_z <= -1.5 or vol_z >= 1.5:
            csp_state = "CSP_NO"
        if vol_z <= -2.0 and not (dex_z >= 0 and gex_z >= 0):
            csp_state = "CSP_NO"

        cc_state = "CC_OK"
        if pd.isna(dex_z) or pd.isna(gex_z) or pd.isna(vol_z) or pd.isna(skew_z):
            cc_state = "CC_CONDITIONAL"
        if dex_z >= 1.5 or vol_z >= 1.5:
            cc_state = "CC_NO"
        elif gex_z <= -1.5:
            cc_state = "CC_CONDITIONAL"

        if skew_z <= -1.5:
            if csp_state == "CSP_OK":
                csp_state = "CSP_CONDITIONAL"
            elif csp_state == "CSP_CONDITIONAL":
                csp_state = "CSP_NO"
        if skew_z >= 0.75 and csp_state == "CSP_CONDITIONAL":
            csp_state = "CSP_OK"

        if skew_z >= 1.5:
            if cc_state == "CC_OK":
                cc_state = "CC_CONDITIONAL"
            elif cc_state == "CC_CONDITIONAL":
                cc_state = "CC_NO"
        if skew_z <= -0.75 and cc_state == "CC_CONDITIONAL":
            cc_state = "CC_OK"

        csp_flag.append(csp_state)
        cc_flag.append(cc_state)

    result["CSP_flag"] = csp_flag
    result["CC_flag"] = cc_flag
    return result


def add_diagnostics(df: pd.DataFrame, window_percent: float | None) -> pd.DataFrame:
    """Add totals/diagnostic columns per expiry group."""
    result = df.copy()
    result["DEX_total"] = result["DEX"].sum()
    result["NET_GEX_total"] = result["GEX"].sum()
    result["GEX_SKEW_total"] = result["GEX_SKEW"].sum()
    result["VEGA_EXPO_total"] = result["VEGA_EXPO"].sum()
    result["eps"] = np.maximum(0.0001 * result["spot"], 0.01)
    result["M"] = M
    result["window_percent"] = window_percent if window_percent is not None else np.nan
    return result


def sort_output(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Sort output by strike or proximity to spot."""
    if sort_by == "proximity":
        proximity = (df["strike"] - df["spot"]).abs()
        return df.assign(_proximity=proximity).sort_values(by=["_proximity", "strike"]).drop(columns=["_proximity"])
    return df.sort_values(by="strike")


def compute_scores(df: pd.DataFrame, window_percent: float | None, include_flags: bool,
                   include_diagnostics: bool, sort_by: str) -> pd.DataFrame:
    """Compute scores, grouped by expiry if present."""
    df = resolve_columns(df)
    df = validate_inputs(df)

    if "expiry" in df.columns:
        grouped = []
        for expiry, group in df.groupby("expiry", dropna=False):
            scored = compute_group_scores(group, window_percent)
            if include_flags:
                scored = apply_flags(scored)
            if include_diagnostics:
                scored = add_diagnostics(scored, window_percent)
            scored["expiry"] = expiry
            grouped.append(scored)
        output = pd.concat(grouped, ignore_index=True)
    else:
        output = compute_group_scores(df, window_percent)
        if include_flags:
            output = apply_flags(output)
        if include_diagnostics:
            output = add_diagnostics(output, window_percent)

    output["VOL_SHOCK_source"] = "vega_x_oi_proxy"
    output = sort_output(output, sort_by)
    return output


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select required output columns."""
    columns = []
    if "expiry" in df.columns:
        columns.append("expiry")
    columns.extend(
        [
            "spot",
            "strike",
            "DEX_z",
            "GEX_z",
            "GEX_SKEW_z",
            "VOL_SHOCK_z",
            "DEX_meaning",
            "DEX_CSP_action",
            "DEX_CC_action",
            "DEX_Equity_action",
            "GEX_meaning",
            "GEX_CSP_action",
            "GEX_CC_action",
            "GEX_Equity_action",
            "GEX_SKEW_meaning",
            "GEX_SKEW_CSP_action",
            "GEX_SKEW_CC_action",
            "GEX_SKEW_Equity_action",
            "VOL_SHOCK_meaning",
            "VOL_SHOCK_CSP_action",
            "VOL_SHOCK_CC_action",
            "VOL_SHOCK_Equity_action",
        ]
    )

    optional_cols = [
        "CSP_flag",
        "CC_flag",
        "DEX_total",
        "NET_GEX_total",
        "GEX_SKEW_total",
        "VEGA_EXPO_total",
        "eps",
        "M",
        "window_percent",
        "VOL_SHOCK_source",
    ]
    for col in optional_cols:
        if col in df.columns:
            columns.append(col)

    return df.loc[:, columns]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute weekly strike scoring z-scores.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (single expiry or multiple expiries).",
    )
    parser.add_argument(
        "--output",
        default="weekly_strike_scores.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to output JSON file.",
    )
    parser.add_argument(
        "--window-percent",
        type=float,
        default=None,
        help="Optional strike window (e.g., 0.02 for ±2%% of spot).",
    )
    parser.add_argument(
        "--include-flags",
        action="store_true",
        help="Include combined CSP/CC flags in output.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include totals/diagnostic columns in output.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["strike", "proximity"],
        default="strike",
        help="Sort output by strike or proximity to spot.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI usage."""
    configure_logging()
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    scored = compute_scores(
        df,
        window_percent=args.window_percent,
        include_flags=args.include_flags,
        include_diagnostics=args.include_diagnostics,
        sort_by=args.sort_by,
    )
    output_df = build_output(scored)

    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(output_df), output_path)

    if args.output_json:
        json_path = Path(args.output_json)
        output_df.to_json(json_path, orient="records")
        LOGGER.info("Wrote JSON output to %s", json_path)


if __name__ == "__main__":
    main()
