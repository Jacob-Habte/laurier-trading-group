# Laurier Trading Group Research

Quantitative market research, strategy development, and backtesting projects.

## Current project

### Futures Skew Analysis

This project tests whether the skewness of historical futures returns contains a systematic trading signal. The strategy takes long positions in markets with negative skew and short positions in markets with positive skew, then compares the resulting portfolio with a conventional 12-month trend-following benchmark.

The analysis uses a frozen CME futures dataset ending March 28, 2024. It is a historical research project rather than a live trading system.

## Reported backtest results

| Strategy | Annualized return | Volatility | Sharpe ratio | Maximum drawdown | Positive years |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original skew strategy | 5.47% | 9.81% | 0.557 | -23.84% | 68.57% |
| Skew strategy with kill switch | 2.99% | 7.97% | 0.375 | -26.02% | 48.57% |

The original strategy remained active throughout the sample. The kill-switch version was active 64.8% of the time and triggered 12 times.

## Repository structure

```text
futures-skew-analysis/
├── notebooks/        # Research and exploratory analysis
├── src/              # Reusable strategy and testing code
├── data/             # Small, reproducible inputs or download instructions
├── outputs/          # Selected figures and result tables
├── README.md         # Methodology and reproduction guide
└── requirements.txt  # Python dependencies
```

The final structure may vary slightly to preserve the original analysis while making the workflow reproducible.

## Disclaimer

This repository is provided for research and educational purposes only. Historical backtest results do not represent live performance and are not investment advice.
