# Stock Screener Project - Detailed Documentation

This document provides a comprehensive overview of the stock screening project, including file descriptions, analysis, and recommendations for which files are redundant or superseded.

## Quick Start Commands

For the best results with detailed stage counts and proper ranking:

```
cd growth_stock_screener
run_modified_screen.bat
```

For a simpler run that still produces good results:

```
cd growth_stock_screener
perfect_setup_direct.bat
```

To fix issues with existing results:

```
cd growth_stock_screener
fix_trading_levels.bat
```

## Project Structure and File Analysis

### Core Screening Files

| File | Purpose | Status |
|------|---------|--------|
| `growth_stock_screener/custom_screen.py` | Original stock screening script with multiple filters | Active, but has issues |
| `growth_stock_screener/custom_screen_modified.py` | Improved version that shows stage counts and fixes issues | **Recommended** |
| `growth_stock_screener/fix_trading_levels.py` | Fixes negative prices and trading levels, adds ranking | Active |
| `growth_stock_screener/perfect_setup_complete.py` | Comprehensive script combining all improvements | Active |
| `growth_stock_screener/perfect_setup_ranked.py` | Runs custom_screen.py and adds ranking | Superseded by newer scripts |

### Batch Files

| File | Purpose | Status |
|------|---------|--------|
| `growth_stock_screener/run_modified_screen.bat` | Runs the modified screen with proper parameters | **Recommended** |
| `growth_stock_screener/perfect_setup_direct.bat` | Directly runs custom_screen.py with correct parameters | Active |
| `growth_stock_screener/fix_trading_levels.bat` | Fixes issues in existing results | Active |
| `growth_stock_screener/perfect_setup_complete.bat` | Runs the complete script | Superseded by run_modified_screen.bat |
| `growth_stock_screener/perfect_setup_ranked.bat` | Runs the ranked script | Superseded by newer batch files |

### Technical Analysis Modules

| File | Purpose | Status |
|------|---------|--------|
| `growth_stock_screener/screen/iterations/technical_indicators.py` | RSI and other technical indicators | Active |
| `growth_stock_screener/screen/iterations/technical_patterns.py` | Pattern detection (channels, etc.) | Active |
| `growth_stock_screener/screen/iterations/utils/` | Utility functions | Active |

### Other Files

| File | Purpose | Status |
|------|---------|--------|
| `growth_stock_screener/screen/settings.py` | Default settings for screening | Active |
| `growth_stock_screener/test_rsi.py` | Test script for RSI calculations | Can be deleted |
| `growth_stock_screener/setup_environment.py` | Sets up the environment | Can be deleted if setup is complete |
| `growth_stock_screener/run_screen.py` | Original run script | Superseded by batch files |
| `growth_stock_screener/run_screen_modified.py` | Modified run script | Superseded by batch files |

## Analysis of Redundant or Superseded Files

### Files That Can Be Deleted

1. **test_rsi.py** - Just a test script, not needed for production
2. **setup_environment.py** - Only needed for initial setup
3. **run_screen.py** - Superseded by the batch files
4. **run_screen_modified.py** - Superseded by the batch files
5. **perfect_setup_ranked.py** - Superseded by perfect_setup_complete.py and custom_screen_modified.py
6. **perfect_setup_ranked.bat** - Superseded by run_modified_screen.bat and perfect_setup_direct.bat
7. **perfect_setup_complete.bat** - Superseded by run_modified_screen.bat

### Files to Keep

1. **custom_screen.py** - Core functionality that other scripts depend on
2. **custom_screen_modified.py** - Improved version with stage counts
3. **fix_trading_levels.py** - Useful utility for fixing issues
4. **perfect_setup_complete.py** - Comprehensive solution
5. **run_modified_screen.bat** - Main entry point for users
6. **perfect_setup_direct.bat** - Alternative entry point
7. **fix_trading_levels.bat** - Utility batch file

## Recommended Workflow

1. Use `run_modified_screen.bat` as the primary tool for screening
2. If any issues occur, use `fix_trading_levels.bat` to fix the results
3. For a simpler run, use `perfect_setup_direct.bat`

## Issues Addressed in the Latest Scripts

1. **SettingWithCopyWarning**:
   - Fixed by creating proper copies of DataFrames before modifying them
   - Used DataFrame merges instead of iterating through rows for better performance

2. **Negative Prices and Trading Levels**:
   - Scripts now detect and fix negative or zero values for:
     - Current prices
     - Ideal entry prices
     - Target prices
     - Stop loss levels
   - Recalculates risk-reward ratios based on fixed values

3. **Default Parameters Issue**:
   - The original script was using default parameters (market cap min: $1B, price min: $10) even when different values were specified on the command line
   - New scripts explicitly set all parameters to the correct values
   - Disables all liquidity filters (market cap, price) by setting them to 0
   - Keeps only the volume filter (1M+ shares) and monthly RSI filter (<25)

4. **Ranking System**:
   - Ranks stocks based on a composite score that considers:
     - Channel Score (25% weight)
     - Perfect Setup Score (25% weight)
     - Risk-Reward Ratio (25% weight)
     - RSI values (15% weight)
     - Trading volume (10% weight)
   - Assigns ratings: STRONG BUY, BUY, NEUTRAL, WEAK, AVOID
   - Clearly shows the #1 ranked stock as the best buy candidate

5. **Stage Counts**:
   - Shows the count of stocks at each filtering stage
   - Makes it clear how many stocks are being filtered out at each step

## Syncing to GitHub

To sync the solution to GitHub, use the following commands:

```
git add .
git commit -m "Improved stock screener with fixed trading levels and ranking"
git push origin main
```

Or if you're using a different branch:

```
git add .
git commit -m "Improved stock screener with fixed trading levels and ranking"
git push origin your-branch-name
```
