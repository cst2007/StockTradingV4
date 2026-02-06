# StockTradingV4 — Full Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Flow](#3-data-flow)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Input CSV Formats](#5-input-csv-formats)
6. [Exposure Formulas](#6-exposure-formulas)
7. [Z-Scores](#7-z-scores)
8. [Decision Tables](#8-decision-tables)
9. [Output Column Reference](#9-output-column-reference)
10. [Running the Application](#10-running-the-application)

---

## 1. Project Overview

StockTradingV4 is a stock options data processor. It takes two broker-exported CSVs per ticker — a side-by-side options file and a Greeks file — merges them on strike price, calculates dealer-positioning exposure metrics, computes cross-strike z-scores, and produces per-strike trading recommendations for Cash Secured Puts (CSP) and Covered Calls (CC).

**Stack:** Python 3 · Flask · pandas · numpy

**Core idea:** Market makers (dealers) are SHORT options. Every formula applies a negative sign to represent dealer hedging flow. Z-scores measure how statistically significant each strike is relative to the others in the same expiry window. The decision tables translate those z-scores into actionable CSP/CC verdicts.

---

## 2. Directory Structure

```
StockTradingV4/
│
├── input/                          Raw broker CSVs (drop files here)
│   └── processed/                  Archived files after processing
│
├── output/
│   ├── processing/                 Stage 1 output: merged unified CSVs
│   └── base_calculations/          Stage 2 output: exposures + z-scores + decisions
│
├── templates/
│   ├── index.html                  Pair-processing UI
│   └── results.html                Decision-table analysis UI
│
├── app.py                          Flask routes and full pipeline orchestration
├── csv_processor.py                Discovery, validation, and merge logic
├── generate_base_calculations.py   Exposures, z-scores, decisions, rankings
├── weekly_strike_scores.py         Standalone CLI scorer (legacy)
├── diagnose_pair.py                Debug utility for troubleshooting pairs
├── normalize_options_files.py      Standalone decimal normaliser (legacy)
└── requirements.txt                pandas, flask, numpy
```

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER                                                               │
│  1. Drops two CSVs into input/  (one side-by-side + one Greeks)     │
│  2. Opens browser → localhost:5000                                  │
│  3. Enters Spot price, clicks "Process Pair"                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  POST /process_pair
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — DISCOVERY  (csv_processor.py › discover_pairs)          │
│                                                                     │
│  Scans input/*.csv against two regex patterns:                      │
│    OPTIONS:  <ticker>-options-exp-<expiry>-…-side-by-side-<date>    │
│    GREEKS:   <ticker>-volatility-greeks-exp-<expiry>-…-<date>       │
│                                                                     │
│  Groups files by (ticker, expiry, run_date) → FilePair objects.     │
│  Incomplete sets (missing one file) are warned and skipped.         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — VALIDATE  (csv_processor.py › validate_pair)            │
│                                                                     │
│  For each file checks:                                              │
│    • File is readable (utf-8-sig encoding, handles BOM)             │
│    • Header row is non-empty                                        │
│    • "Strike" column exists (case-insensitive, any position)        │
│    • Side file has: Volume, Open Int/Interest, IV                   │
│    • Greeks file has: Delta, Gamma, Theta                           │
│                                                                     │
│  Returns (bool, error_string). Error is sent to the UI on failure. │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — MERGE  (csv_processor.py › merge_pair)                  │
│                                                                     │
│  load_side_df():                                                    │
│    • Reads CSV with actual headers (no positional assumption)       │
│    • Locates Strike → splits left (calls) / right (puts)            │
│    • Maps columns by name using strip_pandas_suffix() to handle     │
│      pandas' .1/.2 dedup on duplicate column names                  │
│    • Extracts: Strike, call/put Volume, OI, IV                      │
│                                                                     │
│  load_greeks_df():                                                  │
│    • Same split-on-Strike approach                                  │
│    • Extracts: Strike, call/put Delta, Gamma, Theta, Vega, IV       │
│    • Vega and IV default to 0 if absent                             │
│                                                                     │
│  pd.merge(greeks, side, on="Strike", how="inner")                   │
│    • Greeks IV takes priority; side IV is fallback                   │
│    • Adds metadata: Symbol, Date, Expiry, Spot                      │
│                                                                     │
│  Output: options_unified_<TICKER>_<DATE>.csv  (21 columns)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  Spot value injected here
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — CALCULATE EXPOSURES                                      │
│           (generate_base_calculations.py › calculate_exposures)      │
│                                                                     │
│  Applies DEX, GEX, GEX_SKEW, VOL_SHOCK, THETA_EXPO formulas.       │
│  See Section 6 (Exposure Formulas) for full detail.                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — WINDOW TOTALS                                            │
│           (generate_base_calculations.py › calculate_window_totals)  │
│                                                                     │
│  Sums each exposure across all strikes in the expiry window.        │
│  Stored as metadata columns (same value on every row).              │
│  Key totals: DEX_total, DEX_$_total, NET_GEX_total, GEX_SKEW_total, │
│    VEGA_EXPO_total, THETA_EXPO_total, Total_Call_OI, Total_Put_OI   │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Z-SCORES                                                 │
│           (generate_base_calculations.py › calculate_z_scores)       │
│                                                                     │
│  z = (value − mean) / std   across all strikes in the window.       │
│  Metrics scored: DEX, GEX, GEX_SKEW, VOL_SHOCK, THETA_EXPO,        │
│                  IVxOI, Avg_IV                                      │
│  Each z is classified into one of 6 bands.                          │
│  See Section 7 (Z-Scores) for full detail.                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — DECISION TABLES                                          │
│           (generate_base_calculations.py › add_decision_tables)      │
│                                                                     │
│  Per-metric CSP/CC actions → Hard Blocks → Combined Signals.        │
│  See Section 8 (Decision Tables) for full detail.                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 7 — RANKINGS  (add_rankings)                                 │
│                                                                     │
│  Each metric ranked by absolute value across all strikes.           │
│  Rank 1 = largest absolute value.                                   │
│  Ranked metrics: DEX, GEX, GEX_SKEW, VOL_SHOCK, THETA_EXPO,       │
│                  IVxOI, OI_Imbalance                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 8 — WRITE OUTPUT                                             │
│                                                                     │
│  output/processing/                                                 │
│    options_unified_<TICKER>_<DATE>.csv     (21 columns)             │
│                                                                     │
│  output/base_calculations/                                          │
│    base_calculations_<TICKER>_<DATE>.csv   (80+ columns)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  USER → GET /results                                                │
│                                                                     │
│  Loads base_calculations files via GET /api/calculation/<filename>  │
│  Displays: summary cards, window totals, 4-tab strike table         │
│    Tabs: Overview · CSP Analysis · CC Analysis · Detailed Metrics   │
│  Filters: All · CSP Strong · CC Strong · Blocked · Neutral Zone     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. File-by-File Reference

### csv_processor.py — Discovery, Validation, Merge

| Component | What it does |
|-----------|--------------|
| `OPTIONS_PATTERN` / `GREEKS_PATTERN` | Regex extracting ticker, expiry, run_date from filenames |
| `normalize_ticker()` | Uppercases ticker; maps any SPX variant → `$SPX` |
| `discover_pairs()` | Walks `input/*.csv`, matches regexes, groups into `FilePair` objects |
| `validate_headers()` | Returns `(bool, error_str)`. Checks Strike exists, then checks domain-specific required columns |
| `read_header()` | Opens with `utf-8-sig` (strips BOM) and reads row 0 |
| `strip_pandas_suffix()` | `re.sub(r'\.\d+$', '', name)` — strips `.1`, `.2` pandas adds to duplicate column names |
| `find_strike_index()` | Case-insensitive search for `"strike"` in headers |
| `load_side_df()` | Splits on Strike. Maps call/put columns by name. Extracts Volume, OI, IV |
| `load_greeks_df()` | Same split approach. Extracts Delta, Gamma, Theta, Vega, IV. Vega/IV default to 0 if absent |
| `merge_pair()` | Inner join on Strike. Combines IV sources. Adds metadata. Selects OUTPUT_COLUMNS |
| `validate_pair()` | Calls validate_headers on both files. Returns `(bool, combined_error_string)` |

### generate_base_calculations.py — Calculations & Decisions

| Component | What it does |
|-----------|--------------|
| `calculate_moneyness_signs()` | Returns `(sign_call, sign_put)` arrays: OTM=+1, ITM=−1, ATM=0 |
| `calculate_exposures()` | All 5 exposure formulas using M=100 and negative dealer sign |
| `calculate_window_totals()` | Sums per-strike metrics into a window-wide totals dict |
| `calculate_z_scores()` | `(value − mean) / std` for 7 metrics |
| `classify_z_band()` | Maps float z-score → one of 6 band strings |
| `add_decision_tables()` | 8 per-metric action functions + 2 hard-block functions + 2 combined-signal functions |
| `add_rankings()` | Ranks 7 metrics by absolute value (1 = largest) |

### app.py — Flask Web Layer

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Discovers pairs, renders index.html with pair cards |
| `/process_pair` | POST | Full pipeline: validate → merge → inject spot → exposures → totals → z-scores → decisions → rankings → write |
| `/results` | GET | Lists base_calculations files, renders results.html |
| `/api/calculation/<filename>` | GET | Returns JSON: strike data, window totals, summary counts |

### templates/index.html — Processing UI

- One card per discovered pair: ticker, expiry, run date, both filenames
- Spot price input field per card
- `processPair()` JS POSTs to `/process_pair`, displays success or error inline
- Nav link to `/results`

### templates/results.html — Analysis UI

- File dropdown populated from `/results`
- Filter buttons: All · CSP Strong · CC Strong · Blocked · Neutral Zone
- Summary cards: symbol/date, CSP strong count, CC strong count, blocked count
- Window Totals grid
- 4 tabs: Overview · CSP Analysis · CC Analysis · Detailed Metrics
- Color coding on z-scores: red (≤−2) · yellow (−2 to −0.75) · green (neutral) · blue (+0.75 to +1.5) · purple (≥+1.5)
- Signal badges: green STRONG · light-green GOOD · blue OK · orange CONDITIONAL · red NO

---

## 5. Input CSV Formats

### Side-by-side file

Calls are left of Strike, puts are right. The processor does not require any specific column count or order — it locates Strike, then treats everything to its left as calls and everything to its right as puts.

```
Type,Last,Bid,Ask,Change,Volume,Open Int,IV,Last Trade,Strike,Type,Last,Bid,Ask,Change,Volume,Open Int,IV,Last Trade
Call,13.02,12.70,14.50,+0.67,16,71,27.95%,11/28/25,210.00,Put,1.13,0.85,1.50,-0.62,108,40,47.24%,11/28/25
```

Required columns (by name, case-insensitive): **Strike**, **Volume**, **Open Int** (or Open Interest), **IV**

### Greeks file

Same layout: calls left of Strike, puts right.

```
Last,Theor.,IV,Delta,Gamma,Theta,Vega,Last Trade,Strike,Last,Theor.,IV,Delta,Gamma,Theta,Vega,Last Trade
13.02,13.02,27.95%,0.9533,0.0122,-0.0861,0.0279,11/28/25,210.00,1.13,1.13,47.24%,-0.1559,0.0177,-0.2654,0.0683,11/28/25
```

Required columns: **Strike**, **Delta**, **Gamma**, **Theta**
Optional columns: **Vega**, **IV** (default to 0 if absent)

### Filename conventions

```
<ticker>-options-exp-<YYYY-MM-DD>-<anything>-side-by-side-<MM-DD-YYYY>.csv
<ticker>-volatility-greeks-exp-<YYYY-MM-DD>-<anything>-<MM-DD-YYYY>.csv
```

The ticker, expiry, and run_date extracted from the filename are used to match pairs. The `<anything>` portion can contain any text (e.g. `weekly-20-strikes-+_--`).

---

## 6. Exposure Formulas

All formulas use **M = 100** (standard US options contract multiplier) and a **negative dealer sign** (dealers are short options, so their exposure is the opposite of the open interest holder's).

### DEX — Delta Exposure

Measures directional pressure from dealer delta hedging. No spot scaling.

```
CALL_DEX  = −Δc × OIc × M
PUT_DEX   = −Δp × OIp × M
DEX       = CALL_DEX + PUT_DEX
```

Negative DEX = dealers short delta = downward price pressure.
Positive DEX = dealers long delta = upward price pressure.

### GEX — Gamma Exposure

Uses moneyness sign functions to determine whether dealers are short or long gamma at each strike.

```
Moneyness signs:
  sign_call = +1  if Spot > Strike   (call is OTM, dealers short gamma)
            = −1  if Spot < Strike   (call is ITM, dealers long gamma)
            =  0  if |Spot − Strike| < ε   (ATM neutral band)

  sign_put  = +1  if Spot < Strike   (put is OTM, dealers short gamma)
            = −1  if Spot > Strike   (put is ITM, dealers long gamma)
            =  0  if |Spot − Strike| < ε

  ε = Spot × 0.001   (0.1% of Spot defines the ATM band)

CALL_GEX  = sign_call × Γc × OIc × M
PUT_GEX   = sign_put  × Γp × OIp × M
GEX       = CALL_GEX + PUT_GEX
```

Negative GEX = dealers net short gamma = prices move fast (unstable).
Positive GEX = dealers net long gamma = prices pinned (stable).

### GEX_SKEW — Put/Call Gamma Asymmetry

```
GEX_SKEW = PUT_GEX − CALL_GEX
```

Negative skew = more call-side gamma = resistance dominates.
Positive skew = more put-side gamma = support dominates.

### VOL_SHOCK — Vega Exposure

Measures dealer sensitivity to an implied volatility spike.

```
CALL_VEGA_EXPO = −νc × OIc × M
PUT_VEGA_EXPO  = −νp × OIp × M
VOL_SHOCK      = CALL_VEGA_EXPO + PUT_VEGA_EXPO
```

Large positive VOL_SHOCK at a strike = high vega risk there. An IV spike will hurt dealers most at those strikes, potentially forcing aggressive hedging.

### THETA_EXPO — Time Decay Exposure

```
CALL_THETA_EXPO = −Θc × OIc × M
PUT_THETA_EXPO  = −Θp × OIp × M
THETA_EXPO      = CALL_THETA_EXPO + PUT_THETA_EXPO
```

### Additional Metrics

```
IVxOI            = (Call_IV × OIc) + (Put_IV × OIp)
Distance_to_Spot = |Strike − Spot|
Rel_Dist         = Distance_to_Spot / Spot
OI_Imbalance     = OIc − OIp
Avg_IV           = (Call_IV + Put_IV) / 2
```

---

## 7. Z-Scores

### Calculation

For each metric M across all strikes in one expiry window:

```
z = (M − mean(M)) / std(M)
```

If std = 0 (all strikes have identical values), z is set to 0.

Metrics with z-scores: DEX, GEX, GEX_SKEW, VOL_SHOCK, THETA_EXPO, IVxOI, Avg_IV

### 6 Z-Score Bands

Each z-score is classified into a band label:

```
z ≤ −2.0              →  "≤ −2.0"            (extreme negative)
−2.0 < z ≤ −1.5       →  "−2.0 to −1.5"
−1.5 < z ≤ −0.75      →  "−1.5 to −0.75"
−0.75 < z ≤ +0.75     →  "−0.75 to +0.75"    (neutral zone)
+0.75 < z ≤ +1.5      →  "0.75 to +1.5"
z > +1.5              →  "≥ +1.5"            (extreme positive)
```

Band columns added: `DEX_z_band`, `GEX_z_band`, `GEX_SKEW_z_band`, `VOL_SHOCK_z_band`

---

## 8. Decision Tables

The decision table is built in 4 layers, each feeding into the next. Layers 1 and 2 are informational. Layers 3 and 4 produce the final actionable verdict.

---

### Layer 1 — Per-Metric Actions (Informational)

Each metric independently assigns a CSP label and a CC label for every strike based on its z-score. These are descriptive — they do not block or override anything on their own.

#### DEX_z Actions

DEX measures directional pressure. CSP wants upward pressure; CC wants neutral-to-downward.

| DEX_z Band | DEX_CSP_Action | DEX_CC_Action |
|---|---|---|
| ≤ −2.0 | ❌ Never CSP | ⚠️ CC for protection only |
| −2.0 to −1.5 | ❌ Avoid CSP | ⚠️ CC acceptable at resistance |
| −1.5 to −0.75 | ⚠️ CSP only with strong GEX_z | ✅ CC attractive |
| −0.75 to +0.75 | ✅ Ideal CSP zone | ✅ Ideal CC zone |
| 0.75 to +1.5 | ✅ Strong CSP | ⚠️ CC only at resistance |
| ≥ +1.5 | ✅ Best CSP | ❌ Avoid CC |

#### GEX_z Actions

GEX measures gamma stability. Both strategies prefer stability, but CSP benefits more from high positive GEX. Extreme negative GEX is dangerous for both.

| GEX_z Band | GEX_CSP_Action | GEX_CC_Action |
|---|---|---|
| ≤ −2.0 | ❌ Avoid CSP | ⚠️ CC for protection only |
| −2.0 to −1.5 | ❌ Avoid / small size only | ⚠️ Wait / protection only |
| −1.5 to −0.75 | ⚠️ CSP with confirmation | ✅ CC attractive |
| −0.75 to +0.75 | ✅ Normal CSP | ✅ Normal CC |
| 0.75 to +1.5 | ✅ Preferred CSP zone | ⚠️ CC less attractive |
| ≥ +1.5 | ✅ Best CSP | ⚠️ CC only at resistance |

#### GEX_SKEW_z Actions

GEX_SKEW is the only metric where CSP and CC are exact opposites. CSP wants support (positive skew). CC wants resistance (negative skew). The Equity Meaning column translates the skew into directional language.

| GEX_SKEW_z Band | CSP Action | CC Action | Equity Meaning |
|---|---|---|---|
| ≤ −2.0 | ❌ Avoid CSP | ✅ Strongly favors CC | Resistance very strong, support fragile |
| −2.0 to −1.5 | ⚠️ CSP only if DEX_z≥0 & GEX_z>0 | ✅ CC favored | Sell rips / fade strength |
| −1.5 to −0.75 | ⚠️ CSP selective | ✅ CC slightly favored | Resistance > support |
| −0.75 to +0.75 | Neutral | Neutral | No side advantage |
| 0.75 to +1.5 | ✅ CSP favored | ⚠️ CC selective | Support > resistance |
| ≥ +1.5 | ✅ Strongly favors CSP | ❌ Avoid CC | Strong support / dip-buy behavior |

#### VOL_SHOCK_z Actions

Both strategies are hurt by extreme positive VOL_SHOCK (IV shock risk). Extreme negative is mostly fine — just means low vol exposure at that strike.

| VOL_SHOCK_z Band | VOL_SHOCK_CSP_Action | VOL_SHOCK_CC_Action |
|---|---|---|
| ≤ −2.0 | ✅ CSP okay if DEX/GEX ok | ✅ CC okay |
| −2.0 to −1.5 | ✅ CSP okay | ✅ CC okay |
| −1.5 to −0.75 | ✅ Normal | ✅ Normal |
| −0.75 to +0.75 | ✅ Ideal zone | ✅ Ideal zone |
| 0.75 to +1.5 | ⚠️ CSP only if strong support | ⚠️ CC only at resistance |
| ≥ +1.5 | ❌ Avoid CSP | ❌ Avoid CC |

---

### Layer 2 — Hard Blocks

Hard blocks are checked before the combined signal. If a hard block fires, no upgrade condition can override it.

#### CSP Hard Blocks (`CSP_Hard_Blocks` column)

Any ONE of these conditions produces `CSP_NO`:

| Condition | Reason |
|---|---|
| DEX_z ≤ −1.5 | Strong downside pressure |
| GEX_z ≤ −1.5 | Gamma instability — gap risk |
| VOL_SHOCK_z ≥ +1.5 | IV shock risk at this strike |
| VOL_SHOCK_z ≤ −2.0 AND NOT (DEX_z ≥ 0 AND GEX_z ≥ 0) | Extreme low vol without both directional and gamma confirmation |

Output when blocked: `CSP_NO: <reason1>; <reason2>`
Output when clear: `CSP_OK`

#### CC Hard Blocks (`CC_Hard_Blocks` column)

Any ONE of these conditions produces `CC_CONDITIONAL`:

| Condition | Reason |
|---|---|
| DEX_z ≥ +1.5 | Extreme upside suppression — calls get assigned |
| VOL_SHOCK_z ≥ +1.5 | Vol expansion risk |
| GEX_z ≤ −1.5 | Instability — prefer waiting |

Output when blocked: `CC_CONDITIONAL: <reason1>; <reason2>`
Output when clear: `CC_OK`

> **Note the asymmetry:** CSP hard blocks produce a hard NO. CC hard blocks produce CONDITIONAL — the combined signal is capped but not an absolute stop.

---

### Layer 3 — Combined Signals (Final Verdict)

This is the single per-strike verdict. Each strike gets exactly one `CSP_Combined_Signal` and one `CC_Combined_Signal`.

#### CSP Combined Signal — Decision Tree

```
Step 1: Hard Block Check
  ─────────────────────
  IF CSP_Hard_Blocks = "CSP_NO: …"
    → ❌ CSP_NO (Hard Block)
    STOP.

Step 2: Base Criteria Check  (ALL three must pass)
  ──────────────────────────────────────────────
  DEX_z       ≥ −0.75
  GEX_z       >  −0.75
  VOL_SHOCK_z ≤ +0.75

  IF any one fails
    → ⚠️ CSP_CONDITIONAL (Base criteria not met)
    STOP.

Step 3: Upgrade Count  (each checked independently)
  ───────────────────────────────────────────────
  DEX_z ≥ 0                              → +1
  GEX_z ≥ +0.75                          → +1
  VOL_SHOCK_z in [−0.75, +0.75]          → +1

  upgrades ≥ 2  →  ✅✅ CSP_STRONG (Multiple upgrades)
  upgrades = 1  →  ✅  CSP_GOOD  (Some upgrades)
  upgrades = 0  →  ✅  CSP_OK    (Base criteria met)
```

#### CC Combined Signal — Decision Tree

```
Step 1: Hard Block Check
  ─────────────────────
  IF CC_Hard_Blocks = "CC_CONDITIONAL: …"
    → ⚠️ CC_CONDITIONAL (Check blocks)
    STOP.

Step 2: Base Criteria Check  (ALL three must pass)
  ──────────────────────────────────────────────
  DEX_z       ≤ +0.75
  VOL_SHOCK_z ≤ +1.0
  GEX_z       ≥ −0.75

  IF any one fails
    → ⚠️ CC_CONDITIONAL (Base criteria not met)
    STOP.

Step 3: Upgrade Count  (each checked independently)
  ───────────────────────────────────────────────
  DEX_z       ≤  0                       → +1
  VOL_SHOCK_z ≤  0                       → +1
  GEX_z       ≥ +0.75                    → +1

  upgrades ≥ 2  →  ✅✅ CC_STRONG (Multiple upgrades)
  upgrades = 1  →  ✅  CC_GOOD  (Some upgrades)
  upgrades = 0  →  ✅  CC_OK    (Base criteria met)
```

---

### Layer 4 — All Possible Final Outputs

| Signal | Meaning |
|---|---|
| ❌ CSP_NO (Hard Block) | A hard-block condition fired. Do not sell puts at this strike. |
| ⚠️ CSP_CONDITIONAL (Base criteria not met) | Directional or vol conditions are not favorable. Manual confirmation required. |
| ✅ CSP_OK (Base criteria met) | All base criteria pass. Zero upgrade conditions met. Acceptable. |
| ✅ CSP_GOOD (Some upgrades) | Base criteria pass. One upgrade condition met. |
| ✅✅ CSP_STRONG (Multiple upgrades) | Base criteria pass. Two or more upgrades met. Highest conviction. |
| ⚠️ CC_CONDITIONAL (Check blocks) | A CC hard-block fired. Review the specific reason before entering. |
| ⚠️ CC_CONDITIONAL (Base criteria not met) | Directional or vol conditions not favorable. |
| ✅ CC_OK (Base criteria met) | All base criteria pass. Zero upgrades. Acceptable. |
| ✅ CC_GOOD (Some upgrades) | Base criteria pass. One upgrade met. |
| ✅✅ CC_STRONG (Multiple upgrades) | Base criteria pass. Two or more upgrades met. Highest conviction. |

---

### Complete Decision Flow — Visual

```
              raw z-scores
    (DEX_z, GEX_z, GEX_SKEW_z, VOL_SHOCK_z)
                    │
      ┌─────────────┼─────────────────┐
      ▼             ▼                 ▼
 ┌──────────┐ ┌───────────┐   ┌───────────┐
 │ z-band   │ │ per-metric│   │ per-metric│
 │ labels   │ │ CSP labels│   │ CC labels │
 │ (4 cols) │ │ (4 cols)  │   │ (4 cols)  │
 └──────────┘ └───────────┘   └───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Hard Block     │
          │  CSP_Hard_Blocks│
          │  CC_Hard_Blocks │
          └────────┬────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  ┌──────────┐ ┌────────┐ ┌─────────┐
  │ blocked? │ │base ok?│ │ count   │
  │  → NO   │ │ → COND │ │ upgrades│
  └──────────┘ └────────┘ │ → OK /  │
                           │  GOOD / │
                           │  STRONG │
                           └─────────┘
                   │           │
                   ▼           ▼
         CSP_Combined_Signal
         CC_Combined_Signal
```

---

## 8A. Equity Interpretations (Support/Resistance Analysis)

In addition to the CSP/CC options strategy recommendations, the system provides equity-focused interpretations for directional traders. These interpret the same z-scores through the lens of support/resistance behavior, breakout probability, and move characteristics.

### Per-Metric Equity Interpretations

Each metric gets two equity columns: an interpretation and a specific behavior/expectation.

#### DEX_z — Support/Resistance Strength

| Column | Description |
|---|---|
| `DEX_Equity_Interp` | Explains what the DEX_z level means for directional pressure |
| `DEX_Support_Resistance` | Assessment of support/resistance reliability |

**Interpretation table:**

| DEX_z Band | Equity Interpretation | Support/Resistance Behavior |
|---|---|---|
| ≤ −2.0 | Strong downside acceleration | ❌ Support likely to fail hard |
| −2.0 to −1.5 | Downside pressure dominant | ⚠️ Weak / temporary support |
| −1.5 to −0.75 | Mild downside bias | ⚠️ Support needs confirmation |
| −0.75 to +0.75 | Neutral pressure | ✅ Normal technical support/resistance |
| 0.75 to +1.5 | Upside capped / absorption | ✅ Strong support, resistance holds |
| ≥ +1.5 | Forced selling into rallies | 🧲 Strong resistance / pin risk |

**Key takeaway:** Support is only trusted when DEX_z ≥ −0.75. Resistance is strongest when DEX_z ≥ +0.75.

#### GEX_z — Stability vs Breakout

| Column | Description |
|---|---|
| `GEX_Equity_Interp` | Stability regime assessment |
| `GEX_Level_Behavior` | How technical levels will behave |

**Interpretation table:**

| GEX_z Band | Equity Interpretation | Level Behavior |
|---|---|---|
| ≤ −2.0 | Extreme instability | ❌ Levels break violently |
| −2.0 to −1.5 | High trend risk | ⚠️ Support/resistance unreliable |
| −1.5 to −0.75 | Trend-friendly | ⚠️ Breakouts more likely |
| −0.75 to +0.75 | Mixed | Normal TA applies |
| 0.75 to +1.5 | Mean reversion | 🧲 Levels act as magnets |
| ≥ +1.5 | Pinning / compression | 🧲🧲 Very strong S/R, chop |

**Key takeaway:** High positive GEX_z = range trading, fade extremes. Negative GEX_z = breakout or trend continuation mode.

#### VOL_SHOCK_z — Move Speed & Failure Mode

| Column | Description |
|---|---|
| `VOL_SHOCK_Equity_Interp` | Volatility regime and its implications |
| `VOL_SHOCK_Expectation` | What to expect at this strike |

**Interpretation table:**

| VOL_SHOCK_z Band | Equity Interpretation | What to Expect |
|---|---|---|
| ≤ −2.0 | IV crush regime | Slow drift, fake breaks |
| −2.0 to −1.5 | Vol contraction | Breaks lack follow-through |
| −1.5 to −0.75 | Mild compression | Controlled moves |
| −0.75 to +0.75 | Neutral | Clean technical reactions |
| 0.75 to +1.5 | Rising vol sensitivity | Whipsaws, fast moves |
| ≥ +1.5 | IV shock risk | ❌ Explosive failure, gaps, slippage |

**Key takeaway:** High VOL_SHOCK_z = don't trust tight stops. Neutral/low VOL_SHOCK_z = clean technical reactions.

### Combined Equity Signal

The `Equity_Combined_Signal` column synthesizes all three z-scores into a single directional verdict. It tells you whether to buy dips, fade rallies, trade momentum, or stay cautious.

| Signal | Conditions | Meaning |
|---|---|---|
| 🟢 Strong Support (Buyable Dip) | DEX_z ≥ −0.75<br>GEX_z ≥ +0.75<br>VOL_SHOCK_z ≤ +0.75 | Expect absorption and mean reversion. Dips are buyable. Strong support level. |
| 🔴 Support Likely to Fail | DEX_z ≤ −1.5<br>**OR** GEX_z ≤ −1.5<br>**OR** VOL_SHOCK_z ≥ +1.5 | Expect breakdown or acceleration. Don't trust support here. Exit longs on break. |
| 🧲 Strong Resistance (Fade Zone) | DEX_z ≥ +0.75<br>GEX_z ≥ +0.75<br>VOL_SHOCK_z ≤ 0 | Rallies stall, pins form. Fade strength. Strong resistance cap. |
| 🚀 Breakout / Trend Zone | GEX_z ≤ −0.75<br>VOL_SHOCK_z ≥ 0<br>\|DEX_z\| > 0.5 | Don't fade — trade momentum. Levels less reliable, trend continuation likely. |
| 🟡 Weak / Conditional Support | DEX_z slightly negative<br>GEX_z neutral<br>VOL_SHOCK_z rising | Needs confirmation from volume, time, or structure. Not reliable standalone. |
| ⚪ Neutral Zone | None of the above | No strong directional edge. Wait for better setup. |

**Usage tips:**
- **🟢 Strong Support** — Enter longs on dips to this strike. Set stops below.
- **🔴 Support Fail** — Don't catch the falling knife. Wait for stabilization.
- **🧲 Strong Resistance** — Sell into rallies. Fade rips to this strike.
- **🚀 Breakout Zone** — Ride the momentum. Trail stops, don't fade.
- **🟡 Weak Support** — Only trade if you have additional confluence (higher timeframe, volume spike, candlestick pattern).
- **⚪ Neutral** — No edge. Skip this strike for directional trades.

---

## 9. Output Column Reference

### options_unified_*.csv (21 columns)

The merged raw data before any calculations are applied.

| Column | Description |
|---|---|
| Symbol | Ticker symbol |
| Date | Run date (YYYY-MM-DD) |
| Expiry | Option expiration date |
| Spot | Underlying spot price (entered by user) |
| Strike | Strike price |
| call_delta | Call option delta |
| call_gamma | Call option gamma |
| call_theta | Call option theta |
| call_vega | Call option vega |
| call_open_interest | Call open interest |
| call_volume | Call volume |
| Call_IV | Call implied volatility (decimal) |
| Call_Vanna | Call vanna (reserved) |
| puts_delta | Put option delta |
| put_gamma | Put option gamma |
| put_theta | Put option theta |
| put_vega | Put option vega |
| puts_open_interest | Put open interest |
| put_volume | Put volume |
| Put_IV | Put implied volatility (decimal) |
| Put_Vanna | Put vanna (reserved) |

### base_calculations_*.csv (90+ columns)

Full pipeline output. Includes all columns from the unified file plus:

| Group | Columns |
|---|---|
| DEX | CALL_DEX, PUT_DEX, DEX |
| GEX | CALL_GEX, PUT_GEX, GEX, GEX_SKEW |
| VOL_SHOCK | CALL_VEGA_EXPO, PUT_VEGA_EXPO, VOL_SHOCK |
| THETA | CALL_THETA_EXPO, PUT_THETA_EXPO, THETA_EXPO |
| Derived | IVxOI, Call_IVxOI, Put_IVxOI, Distance_to_Spot, Rel_Dist, OI_Imbalance, Avg_IV |
| Z-scores | DEX_z, GEX_z, GEX_SKEW_z, VOL_SHOCK_z, THETA_EXPO_z, IVxOI_z, Avg_IV_z |
| Z-bands | DEX_z_band, GEX_z_band, GEX_SKEW_z_band, VOL_SHOCK_z_band |
| Per-metric actions | DEX_CSP_Action, DEX_CC_Action, GEX_CSP_Action, GEX_CC_Action, GEX_SKEW_CSP_Action, GEX_SKEW_CC_Action, GEX_SKEW_Equity_Meaning, VOL_SHOCK_CSP_Action, VOL_SHOCK_CC_Action |
| Hard blocks | CSP_Hard_Blocks, CC_Hard_Blocks |
| Final signals | CSP_Combined_Signal, CC_Combined_Signal |
| Equity interpretations | DEX_Equity_Interp, DEX_Support_Resistance, GEX_Equity_Interp, GEX_Level_Behavior, VOL_SHOCK_Equity_Interp, VOL_SHOCK_Expectation, Equity_Combined_Signal |
| Rankings | DEX_Rank, GEX_Rank, GEX_SKEW_Rank, VOL_SHOCK_Rank, THETA_EXPO_Rank, IVxOI_Rank, OI_Imbalance_Rank |
| Window totals | DEX_total, CALL_DEX_total, PUT_DEX_total, DEX_$_total, CALL_GEX_total, PUT_GEX_total, NET_GEX_total, GEX_SKEW_total, VEGA_EXPO_total, CALL_VEGA_EXPO_total, PUT_VEGA_EXPO_total, THETA_EXPO_total, CALL_THETA_EXPO_total, PUT_THETA_EXPO_total, IVxOI_total, Total_Call_OI, Total_Put_OI, Total_OI |

---

## 10. Running the Application

### Prerequisites

```bash
pip install -r requirements.txt
```

### Start the web UI

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### Workflow

1. Copy your broker CSVs (one side-by-side file + one Greeks file per ticker) into the `input/` folder.
2. Refresh the browser. Each matched pair appears as a card.
3. Enter the current spot price for the underlying into the input field on the card.
4. Click **Process Pair**.
5. On success, the output files appear in `output/processing/` and `output/base_calculations/`.
6. Click **View Decision Tables & Analysis** in the nav bar to see the full results.

### Run the CSV processor standalone (without the web UI)

```bash
python csv_processor.py
```

This discovers, validates, and merges all pairs in `input/` but does not run the calculation stages.

### Run base calculations standalone

```bash
python generate_base_calculations.py
```

This reads from `output/processing/` and writes to `output/base_calculations/`. Spot must already be populated in the unified files (done automatically when using the web UI).
