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


def classify_z_band(z_value: float) -> str:
    """Classify a z-score into a band.

    Args:
        z_value: Z-score value

    Returns:
        Band string
    """
    if pd.isna(z_value):
        return "N/A"
    elif z_value <= -2.0:
        return "≤ −2.0"
    elif z_value <= -1.5:
        return "−2.0 to −1.5"
    elif z_value <= -0.75:
        return "−1.5 to −0.75"
    elif z_value <= 0.75:
        return "−0.75 to +0.75"
    elif z_value <= 1.5:
        return "0.75 to +1.5"
    else:
        return "≥ +1.5"


def add_decision_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Add decision table columns based on z-score bands.

    Implements trading strategy recommendations for Cash Secured Puts (CSP)
    and Covered Calls (CC) based on z-score bands for:
    - DEX_z: Directional pressure
    - GEX_z: Gamma regime
    - GEX_SKEW_z: Put/call asymmetry
    - VOL_SHOCK_z: Volatility sensitivity

    Args:
        df: DataFrame with z-score columns

    Returns:
        DataFrame with added decision columns
    """
    result = df.copy()

    # Add z-score band classifications
    for metric in ['DEX', 'GEX', 'GEX_SKEW', 'VOL_SHOCK']:
        z_col = f'{metric}_z'
        if z_col in result.columns:
            result[f'{metric}_z_band'] = result[z_col].apply(classify_z_band)

    # DEX_z Decision Logic
    def dex_csp_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "❌ Never CSP"
        elif z <= -1.5:
            return "❌ Avoid CSP"
        elif z <= -0.75:
            return "⚠️ CSP only with strong GEX_z"
        elif z <= 0.75:
            return "✅ Ideal CSP zone"
        elif z <= 1.5:
            return "✅ Strong CSP"
        else:
            return "✅ Best CSP"

    def dex_cc_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "⚠️ CC for protection only"
        elif z <= -1.5:
            return "⚠️ CC acceptable at resistance"
        elif z <= -0.75:
            return "✅ CC attractive"
        elif z <= 0.75:
            return "✅ Ideal CC zone"
        elif z <= 1.5:
            return "⚠️ CC only at resistance"
        else:
            return "❌ Avoid CC"

    # GEX_z Decision Logic
    def gex_csp_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "❌ Avoid CSP"
        elif z <= -1.5:
            return "❌ Avoid / small size only"
        elif z <= -0.75:
            return "⚠️ CSP with confirmation"
        elif z <= 0.75:
            return "✅ Normal CSP"
        elif z <= 1.5:
            return "✅ Preferred CSP zone"
        else:
            return "✅ Best CSP"

    def gex_cc_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "⚠️ CC for protection only"
        elif z <= -1.5:
            return "⚠️ Wait / protection only"
        elif z <= -0.75:
            return "✅ CC attractive"
        elif z <= 0.75:
            return "✅ Normal CC"
        elif z <= 1.5:
            return "⚠️ CC less attractive"
        else:
            return "⚠️ CC only at resistance"

    # GEX_SKEW_z Decision Logic
    def gex_skew_csp_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "❌ Avoid CSP"
        elif z <= -1.5:
            return "⚠️ CSP only if DEX_z≥0 & GEX_z>0"
        elif z <= -0.75:
            return "⚠️ CSP selective"
        elif z <= 0.75:
            return "Neutral"
        elif z <= 1.5:
            return "✅ CSP favored"
        else:
            return "✅ Strongly favors CSP"

    def gex_skew_cc_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "✅ Strongly favors CC"
        elif z <= -1.5:
            return "✅ CC favored"
        elif z <= -0.75:
            return "✅ CC slightly favored"
        elif z <= 0.75:
            return "Neutral"
        elif z <= 1.5:
            return "⚠️ CC selective"
        else:
            return "❌ Avoid CC"

    def gex_skew_meaning(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "Resistance very strong, support fragile"
        elif z <= -1.5:
            return "Sell rips / fade strength"
        elif z <= -0.75:
            return "Resistance > support"
        elif z <= 0.75:
            return "No side advantage"
        elif z <= 1.5:
            return "Support > resistance"
        else:
            return "Strong support / dip-buy behavior"

    # VOL_SHOCK_z Decision Logic (same as VEX_z)
    def vol_shock_csp_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "✅ CSP okay if DEX/GEX ok"
        elif z <= -1.5:
            return "✅ CSP okay"
        elif z <= -0.75:
            return "✅ Normal"
        elif z <= 0.75:
            return "✅ Ideal zone"
        elif z <= 1.5:
            return "⚠️ CSP only if strong support"
        else:
            return "❌ Avoid CSP"

    def vol_shock_cc_action(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "✅ CC okay"
        elif z <= -1.5:
            return "✅ CC okay"
        elif z <= -0.75:
            return "✅ Normal"
        elif z <= 0.75:
            return "✅ Ideal zone"
        elif z <= 1.5:
            return "⚠️ CC only at resistance"
        else:
            return "❌ Avoid CC"

    # Apply decision functions
    if 'DEX_z' in result.columns:
        result['DEX_CSP_Action'] = result['DEX_z'].apply(dex_csp_action)
        result['DEX_CC_Action'] = result['DEX_z'].apply(dex_cc_action)

    if 'GEX_z' in result.columns:
        result['GEX_CSP_Action'] = result['GEX_z'].apply(gex_csp_action)
        result['GEX_CC_Action'] = result['GEX_z'].apply(gex_cc_action)

    if 'GEX_SKEW_z' in result.columns:
        result['GEX_SKEW_CSP_Action'] = result['GEX_SKEW_z'].apply(gex_skew_csp_action)
        result['GEX_SKEW_CC_Action'] = result['GEX_SKEW_z'].apply(gex_skew_cc_action)
        result['GEX_SKEW_Equity_Meaning'] = result['GEX_SKEW_z'].apply(gex_skew_meaning)

    if 'VOL_SHOCK_z' in result.columns:
        result['VOL_SHOCK_CSP_Action'] = result['VOL_SHOCK_z'].apply(vol_shock_csp_action)
        result['VOL_SHOCK_CC_Action'] = result['VOL_SHOCK_z'].apply(vol_shock_cc_action)

    # Hard Block Rules
    def csp_hard_blocks(row):
        """Check if CSP should be blocked based on hard rules."""
        blocks = []

        # Check each hard block condition
        if 'DEX_z' in row and not pd.isna(row['DEX_z']):
            if row['DEX_z'] <= -1.5:
                blocks.append("DEX_z ≤ -1.5 (strong downside)")

        if 'GEX_z' in row and not pd.isna(row['GEX_z']):
            if row['GEX_z'] <= -1.5:
                blocks.append("GEX_z ≤ -1.5 (instability)")

        if 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']):
            if row['VOL_SHOCK_z'] >= 1.5:
                blocks.append("VOL_SHOCK_z ≥ +1.5 (IV shock risk)")

            # Special case: VEX extreme-low constraint
            if row['VOL_SHOCK_z'] <= -2.0:
                dex_ok = 'DEX_z' in row and not pd.isna(row['DEX_z']) and row['DEX_z'] >= 0
                gex_ok = 'GEX_z' in row and not pd.isna(row['GEX_z']) and row['GEX_z'] >= 0
                if not (dex_ok and gex_ok):
                    blocks.append("VOL_SHOCK_z ≤ -2.0 without DEX≥0 & GEX≥0")

        if blocks:
            return "CSP_NO: " + "; ".join(blocks)
        else:
            return "CSP_OK"

    def cc_hard_blocks(row):
        """Check if CC should be blocked based on hard rules."""
        blocks = []

        # Check each hard block condition
        if 'DEX_z' in row and not pd.isna(row['DEX_z']):
            if row['DEX_z'] >= 1.5:
                blocks.append("DEX_z ≥ +1.5 (extreme upside suppression)")

        if 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']):
            if row['VOL_SHOCK_z'] >= 1.5:
                blocks.append("VOL_SHOCK_z ≥ +1.5 (vol expansion risk)")

        if 'GEX_z' in row and not pd.isna(row['GEX_z']):
            if row['GEX_z'] <= -1.5:
                blocks.append("GEX_z ≤ -1.5 (prefer waiting)")

        if blocks:
            return "CC_CONDITIONAL: " + "; ".join(blocks)
        else:
            return "CC_OK"

    def csp_combined_signal(row):
        """Combined CSP signal based on all criteria."""
        if row.get('CSP_Hard_Blocks', '').startswith('CSP_NO'):
            return "❌ CSP_NO (Hard Block)"

        # Base requirements
        dex_ok = 'DEX_z' in row and not pd.isna(row['DEX_z']) and row['DEX_z'] >= -0.75
        gex_ok = 'GEX_z' in row and not pd.isna(row['GEX_z']) and row['GEX_z'] > -0.75
        vol_ok = 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']) and row['VOL_SHOCK_z'] <= 0.75

        if not (dex_ok and gex_ok and vol_ok):
            return "⚠️ CSP_CONDITIONAL (Base criteria not met)"

        # Upgrade conditions
        dex_strong = 'DEX_z' in row and not pd.isna(row['DEX_z']) and row['DEX_z'] >= 0
        gex_strong = 'GEX_z' in row and not pd.isna(row['GEX_z']) and row['GEX_z'] >= 0.75
        vol_neutral = 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']) and -0.75 <= row['VOL_SHOCK_z'] <= 0.75

        upgrade_count = sum([dex_strong, gex_strong, vol_neutral])

        if upgrade_count >= 2:
            return "✅✅ CSP_STRONG (Multiple upgrades)"
        elif upgrade_count >= 1:
            return "✅ CSP_GOOD (Some upgrades)"
        else:
            return "✅ CSP_OK (Base criteria met)"

    def cc_combined_signal(row):
        """Combined CC signal based on all criteria."""
        if row.get('CC_Hard_Blocks', '').startswith('CC_CONDITIONAL'):
            return "⚠️ CC_CONDITIONAL (Check blocks)"

        # Base requirements
        dex_ok = 'DEX_z' in row and not pd.isna(row['DEX_z']) and row['DEX_z'] <= 0.75
        vol_ok = 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']) and row['VOL_SHOCK_z'] <= 1.0
        gex_ok = 'GEX_z' in row and not pd.isna(row['GEX_z']) and row['GEX_z'] >= -0.75

        if not (dex_ok and vol_ok and gex_ok):
            return "⚠️ CC_CONDITIONAL (Base criteria not met)"

        # Upgrade conditions
        dex_strong = 'DEX_z' in row and not pd.isna(row['DEX_z']) and row['DEX_z'] <= 0
        vol_strong = 'VOL_SHOCK_z' in row and not pd.isna(row['VOL_SHOCK_z']) and row['VOL_SHOCK_z'] <= 0
        gex_tight = 'GEX_z' in row and not pd.isna(row['GEX_z']) and row['GEX_z'] >= 0.75

        upgrade_count = sum([dex_strong, vol_strong, gex_tight])

        if upgrade_count >= 2:
            return "✅✅ CC_STRONG (Multiple upgrades)"
        elif upgrade_count >= 1:
            return "✅ CC_GOOD (Some upgrades)"
        else:
            return "✅ CC_OK (Base criteria met)"

    # Apply hard block checks
    result['CSP_Hard_Blocks'] = result.apply(csp_hard_blocks, axis=1)
    result['CC_Hard_Blocks'] = result.apply(cc_hard_blocks, axis=1)

    # Apply combined signals
    result['CSP_Combined_Signal'] = result.apply(csp_combined_signal, axis=1)
    result['CC_Combined_Signal'] = result.apply(cc_combined_signal, axis=1)

    return result


def add_equity_interpretations(df: pd.DataFrame) -> pd.DataFrame:
    """Add equity-focused support/resistance interpretations.

    Provides directional trading context based on z-score combinations:
    - DEX_z → Support/Resistance strength
    - GEX_z → Stability vs Breakout regime
    - VOL_SHOCK_z → Move speed and failure mode

    Args:
        df: DataFrame with z-score columns

    Returns:
        DataFrame with equity interpretation columns
    """
    result = df.copy()

    # DEX_z Equity Interpretation
    def dex_equity_interp(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "Strong downside acceleration — support likely to fail hard"
        elif z <= -1.5:
            return "Downside pressure dominant — weak / temporary support"
        elif z <= -0.75:
            return "Mild downside bias — support needs confirmation"
        elif z <= 0.75:
            return "Neutral pressure — normal technical support/resistance"
        elif z <= 1.5:
            return "Upside capped / absorption — strong support, resistance holds"
        else:
            return "Forced selling into rallies — strong resistance / pin risk"

    def dex_support_resistance(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "❌ Support likely to fail hard"
        elif z <= -1.5:
            return "⚠️ Weak / temporary support"
        elif z <= -0.75:
            return "⚠️ Support needs confirmation"
        elif z <= 0.75:
            return "✅ Normal technical support/resistance"
        elif z <= 1.5:
            return "✅ Strong support, resistance holds"
        else:
            return "🧲 Strong resistance / pin risk"

    # GEX_z Equity Interpretation
    def gex_equity_interp(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "Extreme instability — levels break violently"
        elif z <= -1.5:
            return "High trend risk — support/resistance unreliable"
        elif z <= -0.75:
            return "Trend-friendly — breakouts more likely"
        elif z <= 0.75:
            return "Mixed — normal TA applies"
        elif z <= 1.5:
            return "Mean reversion — levels act as magnets"
        else:
            return "Pinning / compression — very strong S/R, chop"

    def gex_level_behavior(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "❌ Levels break violently"
        elif z <= -1.5:
            return "⚠️ Support/resistance unreliable"
        elif z <= -0.75:
            return "⚠️ Breakouts more likely"
        elif z <= 0.75:
            return "Normal TA applies"
        elif z <= 1.5:
            return "🧲 Levels act as magnets"
        else:
            return "🧲🧲 Very strong S/R, chop"

    # VOL_SHOCK_z Equity Interpretation
    def vol_shock_equity_interp(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -2.0:
            return "IV crush regime — slow drift, fake breaks"
        elif z <= -1.5:
            return "Vol contraction — breaks lack follow-through"
        elif z <= -0.75:
            return "Mild compression — controlled moves"
        elif z <= 0.75:
            return "Neutral — clean technical reactions"
        elif z <= 1.5:
            return "Rising vol sensitivity — faster moves, whipsaws"
        else:
            return "IV shock risk — levels can fail explosively"

    def vol_shock_expectation(z):
        if pd.isna(z):
            return "N/A"
        elif z <= -1.5:
            return "Slow moves, levels break cleanly"
        elif z <= -0.75:
            return "Breakouts often stall"
        elif z <= 0.75:
            return "Clean technical reactions"
        elif z <= 1.5:
            return "Whipsaws, fast moves"
        else:
            return "❌ Explosive failure, gaps, slippage"

    # Apply equity interpretation functions
    if 'DEX_z' in result.columns:
        result['DEX_Equity_Interp'] = result['DEX_z'].apply(dex_equity_interp)
        result['DEX_Support_Resistance'] = result['DEX_z'].apply(dex_support_resistance)

    if 'GEX_z' in result.columns:
        result['GEX_Equity_Interp'] = result['GEX_z'].apply(gex_equity_interp)
        result['GEX_Level_Behavior'] = result['GEX_z'].apply(gex_level_behavior)

    if 'VOL_SHOCK_z' in result.columns:
        result['VOL_SHOCK_Equity_Interp'] = result['VOL_SHOCK_z'].apply(vol_shock_equity_interp)
        result['VOL_SHOCK_Expectation'] = result['VOL_SHOCK_z'].apply(vol_shock_expectation)

    # Combined Equity Signals
    def equity_combined_signal(row):
        """Determine combined equity signal based on z-score combinations."""
        dex = row.get('DEX_z')
        gex = row.get('GEX_z')
        vol = row.get('VOL_SHOCK_z')

        if pd.isna(dex) or pd.isna(gex) or pd.isna(vol):
            return "N/A"

        # 🟢 Strong Support (Buyable Dip)
        if dex >= -0.75 and gex >= 0.75 and vol <= 0.75:
            return "🟢 Strong Support (Buyable Dip)"

        # 🔴 Support Likely to Fail
        if dex <= -1.5 or gex <= -1.5 or vol >= 1.5:
            return "🔴 Support Likely to Fail"

        # 🧲 Strong Resistance (Fade Zone)
        if dex >= 0.75 and gex >= 0.75 and vol <= 0:
            return "🧲 Strong Resistance (Fade Zone)"

        # 🚀 Breakout / Trend Zone
        if gex <= -0.75 and vol >= 0:
            # Check DEX alignment (absolute value > 0.5 means directional)
            if abs(dex) > 0.5:
                return "🚀 Breakout / Trend Zone"

        # 🟡 Weak / Conditional Support
        if -0.75 <= dex < 0 and -0.75 < gex < 0.75 and vol > 0:
            return "🟡 Weak / Conditional Support"

        # Default: Neutral zone
        return "⚪ Neutral Zone"

    result['Equity_Combined_Signal'] = result.apply(equity_combined_signal, axis=1)

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

            # Add decision tables based on z-score bands
            result_df = add_decision_tables(result_df)

            # Add equity interpretations for support/resistance analysis
            result_df = add_equity_interpretations(result_df)

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
    LOGGER.info("  - Decision Tables: CSP/CC recommendations based on z-score bands")
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
