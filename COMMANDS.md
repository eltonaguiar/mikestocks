# Stock Screener Commands

## Quick Commands for Stock Analysis

### Recommended: Run Modified Screen with Clear Stage Counts

```
cd growth_stock_screener
run_modified_screen.bat
```

### Alternative: Run Direct Perfect Setup

```
cd growth_stock_screener
perfect_setup_direct.bat
```

### Fix Issues in Existing Results

```
cd growth_stock_screener
fix_trading_levels.bat
```

## Sync to GitHub

```
git add .
git commit -m "Improved stock screener with fixed trading levels and ranking"
git push origin main
```

## What These Commands Do

1. **run_modified_screen.bat**:
   - Shows the count of stocks at each filtering stage
   - Properly disables all liquidity filters (market cap, price)
   - Keeps only the volume filter (1M+ shares) and monthly RSI filter (<25)
   - Ranks stocks with clear ratings (STRONG BUY, BUY, etc.)
   - Identifies the #1 ranked stock as the best buy candidate

2. **perfect_setup_direct.bat**:
   - Directly runs custom_screen.py with correct parameters
   - Fixes any negative values in trading levels
   - Ranks the stocks based on a composite score

3. **fix_trading_levels.bat**:
   - Fixes negative or zero prices
   - Fixes negative or zero trading levels (ideal entry, target price, stop loss)
   - Recalculates risk-reward ratios
   - Ranks stocks based on a composite score
