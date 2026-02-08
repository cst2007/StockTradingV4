# Combined Options & Equity Interpretation Guide

## Overview

All tabs now show **both options trading signals AND equity interpretations** together, giving you a complete picture for each strike.

## Updated Tab Structure

### 1. Overview Tab (10 columns)
**Purpose:** Quick scan of all signals together

| Column | Type | What It Shows |
|--------|------|---------------|
| Strike | Base | Strike price |
| DEX_z | Z-score | Directional exposure z-score (color-coded) |
| GEX_z | Z-score | Gamma exposure z-score (color-coded) |
| SKEW_z | Z-score | Skew z-score (color-coded) |
| VOL_z | Z-score | Vol shock z-score (color-coded) |
| CSP Signal | Options | Cash-secured put trading signal |
| CC Signal | Options | Covered call trading signal |
| **Equity Signal** | **Equity** | **Combined equity signal (🟢🔴🧲🚀🟡⚪)** |
| **Support/Resistance** | **Equity** | **Support/resistance context** |
| **Level Behavior** | **Equity** | **How the level behaves (absorption vs rejection)** |

**Highlighting:** Rows with strong CSP signals, strong equity support (🟢), or breakout signals (🚀) are highlighted.

### 2. Equity Interpretations Tab (11 columns)
**Purpose:** Deep dive into support/resistance analysis

| Column | What It Shows |
|--------|---------------|
| Strike | Strike price |
| DEX_z | DEX z-score with color coding |
| DEX Equity Interpretation | Full DEX directional pressure explanation |
| Support/Resistance | Quick DEX summary with ✅/❌ |
| GEX_z | GEX z-score with color coding |
| GEX Equity Interpretation | Stability vs breakout regime |
| Level Behavior | How the level will behave |
| VOL_z | VOL_SHOCK z-score with color coding |
| VOL Equity Interpretation | Move speed and failure mode |
| Move Expectation | Expected move characteristics |
| **Combined Signal** | **Synthesized trading signal** |

**Highlighting:** Rows with 🟢 (strong support) or 🚀 (breakout) signals.

### 3. CSP Analysis Tab (12 columns)
**Purpose:** Cash-secured put strategy with equity context

| Column | Type | What It Shows |
|--------|------|---------------|
| Strike | Base | Strike price |
| DEX_z | Z-score | DEX z-score |
| DEX Action | Options | DEX-based CSP action (✅/⚠️/❌) |
| GEX_z | Z-score | GEX z-score |
| GEX Action | Options | GEX-based CSP action |
| VOL_z | Z-score | VOL_SHOCK z-score |
| VOL Action | Options | VOL-based CSP action |
| SKEW Action | Options | SKEW-based CSP action |
| Hard Blocks | Options | Any blocking conditions |
| **CSP Signal** | **Options** | **Combined CSP signal** |
| **Equity Signal** | **Equity** | **Combined equity signal (🟢🔴🧲🚀🟡⚪)** |
| **Support/Resistance** | **Equity** | **Support/resistance context** |

**Highlighting:** Rows with strong CSP signals OR strong equity support (🟢).

### 4. CC Analysis Tab (12 columns)
**Purpose:** Covered call strategy with equity context

| Column | Type | What It Shows |
|--------|------|---------------|
| Strike | Base | Strike price |
| DEX_z | Z-score | DEX z-score |
| DEX Action | Options | DEX-based CC action (✅/⚠️/❌) |
| GEX_z | Z-score | GEX z-score |
| GEX Action | Options | GEX-based CC action |
| VOL_z | Z-score | VOL_SHOCK z-score |
| VOL Action | Options | VOL-based CC action |
| SKEW Action | Options | SKEW-based CC action |
| Hard Blocks | Options | Any blocking conditions |
| **CC Signal** | **Options** | **Combined CC signal** |
| **Equity Signal** | **Equity** | **Combined equity signal (🟢🔴🧲🚀🟡⚪)** |
| **Support/Resistance** | **Equity** | **Support/resistance context** |

**Highlighting:** Rows with strong CC signals OR strong equity resistance (🧲).

### 5. Detailed Metrics Tab (10 columns)
**Purpose:** Raw exposure values and Greeks

| Column | What It Shows |
|--------|---------------|
| Strike | Strike price |
| DEX | Raw directional exposure |
| GEX | Raw gamma exposure |
| SKEW | Raw skew value |
| VOL_SHOCK | Raw vol shock value |
| Call OI | Call open interest |
| Put OI | Put open interest |
| OI Imbal | Open interest imbalance |
| Call IV | Call implied volatility |
| Put IV | Put implied volatility |

## How to Use the Combined Views

### For CSP Traders (Selling Puts)

**Look at CSP Analysis Tab:**

1. **Best Setup (Highlighted Rows):**
   - CSP Signal = "STRONG" (green badge)
   - Equity Signal = "🟢 Strong Support (Buyable Dip)"
   - Support/Resistance = "✅ Strong support"

   **Interpretation:** Both options flow AND equity analysis agree this is a strong support level. High conviction CSP.

2. **Conflicting Signals:**
   - CSP Signal = "STRONG" but Equity Signal = "🔴 Support Likely to Fail"

   **Interpretation:** Options flow shows support, but equity context warns of breakdown risk. Reduce position size or skip.

3. **Avoid:**
   - Any row with "❌ CSP_NO" in Hard Blocks column
   - Equity Signal = "🔴 Support Likely to Fail"

### For CC Traders (Selling Calls)

**Look at CC Analysis Tab:**

1. **Best Setup (Highlighted Rows):**
   - CC Signal = "STRONG" (green badge)
   - Equity Signal = "🧲 Strong Resistance (Fade Zone)"
   - Support/Resistance = "Strong resistance"

   **Interpretation:** Both options flow AND equity analysis agree this is a strong resistance level. High conviction CC.

2. **Avoid:**
   - Equity Signal = "🚀 Breakout / Trend Zone" (risk of getting called away)
   - CC Signal contains "NO" or "CONDITIONAL"

### For Equity Traders

**Look at Overview or Equity Interpretations Tab:**

1. **Long Entry (Support):**
   - Equity Signal = "🟢 Strong Support (Buyable Dip)"
   - CSP Signal = "STRONG" confirms options dealers will defend this level
   - Level Behavior = "Mean revert" or "Absorb sells"

2. **Short Entry (Resistance):**
   - Equity Signal = "🧲 Strong Resistance (Fade Zone)"
   - CC Signal = "STRONG" confirms options dealers will cap upside
   - Level Behavior = "Reject rallies" or "Cap upside"

3. **Breakout Trade:**
   - Equity Signal = "🚀 Breakout / Trend Zone"
   - GEX_z < -0.75 (unstable regime, volatility expansion)
   - Both CSP and CC signals should NOT be strong (no defending)

## Emoji Quick Reference

| Emoji | Signal | Best For |
|-------|--------|----------|
| 🟢 | Strong Support (Buyable Dip) | CSP, Long equity |
| 🔴 | Support Likely to Fail | Avoid CSP/long |
| 🧲 | Strong Resistance (Fade Zone) | CC, Short equity |
| 🚀 | Breakout / Trend Zone | Directional plays (calls/puts) |
| 🟡 | Weak / Conditional Support | Cautious CSP |
| ⚪ | Neutral Zone | Normal TA applies |

## Example Scenarios

### Scenario 1: Perfect CSP Setup
```
Strike: $210
CSP Signal: ✅ CSP_STRONG
Equity Signal: 🟢 Strong Support (Buyable Dip)
Support/Resistance: ✅ Strong support — expect buyers to defend
DEX_z: +0.85 (positive, upward pressure)
GEX_z: +1.2 (stable regime, mean reversion)

Action: High conviction CSP. Both options and equity agree on support.
```

### Scenario 2: Conflicting Signals - Avoid
```
Strike: $215
CSP Signal: ✅ CSP_GOOD
Equity Signal: 🔴 Support Likely to Fail
Support/Resistance: ❌ Support likely to fail hard
DEX_z: -1.8 (strong downside pressure)
VOL_SHOCK_z: +2.1 (extreme vol expansion)

Action: Skip. Options flow shows support, but equity warns of breakdown.
```

### Scenario 3: Perfect CC Setup
```
Strike: $220
CC Signal: ✅ CC_STRONG
Equity Signal: 🧲 Strong Resistance (Fade Zone)
Support/Resistance: Strong resistance — expect rejection
DEX_z: +1.5 (upward pressure meets ceiling)
GEX_z: +1.8 (stable, mean reversion to downside)

Action: High conviction CC. Both options and equity agree on resistance.
```

### Scenario 4: Breakout Warning
```
Strike: $225
CSP Signal: ❌ CSP_NO
CC Signal: ⚠️ CC_CONDITIONAL
Equity Signal: 🚀 Breakout / Trend Zone
Level Behavior: Expect clean break and follow-through
GEX_z: -1.5 (unstable regime)

Action: Avoid theta strategies. Consider directional calls/puts instead.
```

## Tips

1. **Always check both columns:** Options signals show dealer positioning, equity signals show price action context.

2. **Highest conviction when both agree:**
   - CSP STRONG + 🟢 Strong Support = Best CSP
   - CC STRONG + 🧲 Strong Resistance = Best CC

3. **Be cautious when they conflict:**
   - CSP STRONG but 🔴 Support Fail = Reduce size
   - CC STRONG but 🚀 Breakout = Risk of assignment

4. **Use the Overview tab** for quick scanning, then drill into specific strategy tabs for details.

5. **Row highlighting helps** identify the best opportunities at a glance.

## Code References

- Overview tab: `templates/results.html:493-511`
- CSP tab: `templates/results.html:536-551`
- CC tab: `templates/results.html:553-568`
- Equity tab: `templates/results.html:513-534`
