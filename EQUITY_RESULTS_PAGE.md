# Equity Interpretations in Results Page

## Overview

The results page now includes a new **"Equity Interpretations"** tab showing support/resistance analysis based on z-score combinations.

## New Tab: Equity Interpretations

### Columns Displayed

| Column | Source | Description |
|--------|--------|-------------|
| Strike | Base data | Strike price |
| DEX_z | Calculated | DEX z-score with color coding |
| DEX Equity Interpretation | `DEX_Equity_Interp` | Full directional pressure explanation |
| Support/Resistance | `DEX_Support_Resistance` | Quick summary with emoji (✅/❌) |
| GEX_z | Calculated | GEX z-score with color coding |
| GEX Equity Interpretation | `GEX_Equity_Interp` | Stability vs breakout regime explanation |
| Level Behavior | `GEX_Level_Behavior` | How the level will behave |
| VOL_z | Calculated | VOL_SHOCK z-score with color coding |
| VOL Equity Interpretation | `VOL_SHOCK_Equity_Interp` | Move speed and failure mode |
| Move Expectation | `VOL_SHOCK_Expectation` | Expected move characteristics |
| **Combined Signal** | `Equity_Combined_Signal` | **Synthesized trading signal** |

## Combined Equity Signals

The Combined Signal column shows one of these signals:

| Signal | Emoji | Meaning | Row Highlight |
|--------|-------|---------|---------------|
| Strong Support (Buyable Dip) | 🟢 | Expect absorption and mean reversion | ✅ Highlighted |
| Support Likely to Fail | 🔴 | Expect breakdown or acceleration | ❌ |
| Strong Resistance (Fade Zone) | 🧲 | Expect rejection at this level | ❌ |
| Breakout / Trend Zone | 🚀 | Expect directional movement | ✅ Highlighted |
| Weak / Conditional Support | 🟡 | Support exists but fragile | ⚠️ |
| Neutral Zone | ⚪ | Normal technical analysis applies | — |

## Updated Overview Tab

The **Overview** tab has been updated to show:
- All z-scores (DEX, GEX, SKEW, VOL)
- CSP and CC signals
- **Equity Combined Signal** (new column, replaces GEX_SKEW_Equity_Meaning)

## Tab Navigation

The results page now has 5 tabs:
1. **Overview** - Quick summary with combined equity signal
2. **Equity Interpretations** ⭐ NEW - Detailed support/resistance analysis
3. **CSP Analysis** - Cash-secured put strategy signals
4. **CC Analysis** - Covered call strategy signals
5. **Detailed Metrics** - Raw exposure metrics and Greeks

## Visual Features

### Row Highlighting
- Rows with 🟢 Strong Support signals are **highlighted in yellow**
- Rows with 🚀 Breakout signals are **highlighted in yellow**
- Makes it easy to spot high-conviction equity setups

### Z-Score Color Coding
Z-scores are color-coded for quick reading:
- **Dark Red** (≤ -2.0): Extreme negative
- **Yellow** (-2.0 to -0.75): Negative
- **Green** (-0.75 to +0.75): Neutral
- **Blue** (+0.75 to +1.5): Positive
- **Purple** (≥ +1.5): Extreme positive

### Text Formatting
- Strike prices: Bold purple
- Interpretations: Small font (11px) for readability
- Key summaries (Support/Resistance, Combined Signal): Bold (600 weight)
- Combined Signal: Larger font (13px, 700 weight) with emoji

## Example Display

```
Strike: $210.00
DEX_z: -0.71 (Yellow/Negative)
DEX Equity Interp: "Neutral pressure — normal technical support/resistance"
Support/Resistance: "✅ Normal technical support/resistance"
GEX_z: -0.71 (Yellow/Negative)
GEX Equity Interp: "Mixed — normal TA applies"
Level Behavior: "Normal TA applies"
VOL_z: -0.71 (Yellow/Negative)
VOL Equity Interp: "Neutral — clean technical reactions"
Move Expectation: "Clean technical reactions"
Combined Signal: "⚪ Neutral Zone"
```

## Usage Tips

### For Support Trading (Long/CSP)
1. Look for 🟢 Strong Support signals
2. Check the "Support/Resistance" column for ✅ confirmation
3. Verify "Level Behavior" shows absorption characteristics
4. Avoid ❌ signals or 🔴 Support Likely to Fail

### For Resistance Trading (Short/CC)
1. Look for 🧲 Strong Resistance signals
2. Check "Level Behavior" for rejection patterns
3. Verify high VOL_SHOCK doesn't indicate breakout risk
4. Combine with CC_Combined_Signal from CC tab

### For Breakout Trading
1. Look for 🚀 Breakout / Trend Zone signals
2. Check that GEX is negative (unstable regime)
3. Verify VOL_SHOCK shows expansion potential
4. Use directional options (calls/puts, not theta)

## Code Reference

- Template: `templates/results.html:546-567` (Equity tab structure)
- Rendering function: `templates/results.html:804-834` (renderEquityTable)
- CSS styles: `templates/results.html:342-368` (equity-signal classes)
- Data source: `output/base_calculations/base_calculations_*.csv`
