# CSV Column Mapping and Filtering

This document shows how we filter CSV input columns to extract only the data needed for analysis.

## Input Files Structure

### 1. Side-by-Side File (19 columns in CSV)
**Example:** `SPXL-options-exp-2025-12-05-test-side-by-side-11-28-2025.csv`

| CSV Columns (19 total) | Extracted? | Internal Name |
|------------------------|-----------|---------------|
| Type (Call) | ❌ No | — |
| Last (Call) | ❌ No | — |
| Bid (Call) | ❌ No | — |
| Ask (Call) | ❌ No | — |
| Change (Call) | ❌ No | — |
| **Volume (Call)** | ✅ **Yes** | **call_volume** |
| **Open Int (Call)** | ✅ **Yes** | **call_open_interest** |
| **IV (Call)** | ✅ **Yes** | **call_iv_raw** |
| Last Trade (Call) | ❌ No | — |
| **Strike** | ✅ **Yes** | **Strike** |
| Type (Put) | ❌ No | — |
| Last (Put) | ❌ No | — |
| Bid (Put) | ❌ No | — |
| Ask (Put) | ❌ No | — |
| Change (Put) | ❌ No | — |
| **Volume (Put)** | ✅ **Yes** | **put_volume** |
| **Open Int (Put)** | ✅ **Yes** | **put_open_interest** |
| **IV (Put)** | ✅ **Yes** | **put_iv_raw** |
| Last Trade (Put) | ❌ No | — |

**Extracted: 7 columns** (discarding 12 unused columns)

### 2. Greeks File (17 columns in CSV)
**Example:** `SPXL-volatility-greeks-exp-2025-12-05-test-11-28-2025.csv`

| CSV Columns (17 total) | Extracted? | Internal Name |
|------------------------|-----------|---------------|
| Last (Call) | ❌ No | — |
| Theor. (Call) | ❌ No | — |
| **IV (Call)** | ✅ **Yes** | **call_iv** |
| **Delta (Call)** | ✅ **Yes** | **call_delta** |
| **Gamma (Call)** | ✅ **Yes** | **call_gamma** |
| **Theta (Call)** | ✅ **Yes** | **call_theta** |
| **Vega (Call)** | ✅ **Yes** | **call_vega** |
| Last Trade (Call) | ❌ No | — |
| **Strike** | ✅ **Yes** | **Strike** |
| Last (Put) | ❌ No | — |
| Theor. (Put) | ❌ No | — |
| **IV (Put)** | ✅ **Yes** | **put_iv** |
| **Delta (Put)** | ✅ **Yes** | **puts_delta** |
| **Gamma (Put)** | ✅ **Yes** | **put_gamma** |
| **Theta (Put)** | ✅ **Yes** | **put_theta** |
| **Vega (Put)** | ✅ **Yes** | **put_vega** |
| Last Trade (Put) | ❌ No | — |

**Extracted: 11 columns** (discarding 6 unused columns)

## Data Flow Through Pipeline

```
CSV Input Files (36 total columns)
    ↓
[load_side_df()] extracts 7 columns
[load_greeks_df()] extracts 11 columns
    ↓
Merge on Strike (17 columns total after merge)
    ↓
Add metadata columns:
  • Symbol (from filename)
  • Date (from filename)
  • Expiry (from filename)
  • Spot (placeholder = NA)
  • Call_Vanna (placeholder = NA)
  • Put_Vanna (placeholder = NA)
    ↓
Combine IV sources:
  • Call_IV = call_iv OR call_iv_raw
  • Put_IV = put_iv OR put_iv_raw
    ↓
Rename for consistency:
  • puts_open_interest = put_open_interest
    ↓
Drop intermediate columns:
  • call_iv
  • put_iv
  • call_iv_raw
  • put_iv_raw
  • put_open_interest
    ↓
Filter to OUTPUT_COLUMNS (21 columns)
    ↓
Output to processing/merged_data/
```

## Final OUTPUT_COLUMNS (21 columns)

These are the **only columns** carried into the exposure calculations:

| Column | Source | Used In |
|--------|--------|---------|
| Symbol | Metadata | Identification |
| Date | Metadata | Identification |
| Expiry | Metadata | Identification |
| Spot | Metadata | Identification |
| Strike | Both files | All calculations |
| call_delta | Greeks | **DEX formula** |
| call_gamma | Greeks | **GEX formula** |
| call_theta | Greeks | **THETA_EXPO formula** |
| call_vega | Greeks | **VOL_SHOCK formula** |
| call_open_interest | Side-by-side | **All exposure formulas** |
| call_volume | Side-by-side | Future use |
| Call_IV | Greeks + Side | IV analysis |
| Call_Vanna | Placeholder | Future use |
| puts_delta | Greeks | **DEX formula** |
| put_gamma | Greeks | **GEX formula** |
| put_theta | Greeks | **THETA_EXPO formula** |
| put_vega | Greeks | **VOL_SHOCK formula** |
| puts_open_interest | Side-by-side | **All exposure formulas** |
| put_volume | Side-by-side | Future use |
| Put_IV | Greeks + Side | IV analysis |
| Put_Vanna | Placeholder | Future use |

## Efficiency Summary

- **Input:** 36 CSV columns (19 + 17)
- **Filtered to:** 21 columns (58% reduction)
- **Unused columns discarded:** 18 columns
- **Memory saved:** ~42% by not loading unnecessary data

## Code References

- Column extraction: `csv_processor.py:550-727` (load_side_df, load_greeks_df)
- OUTPUT_COLUMNS definition: `csv_processor.py:120-142`
- Filtering applied: `csv_processor.py:780` (`merged = merged[OUTPUT_COLUMNS]`)
