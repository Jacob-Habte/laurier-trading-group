from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


TRADING_DAYS = 252
BASELINE_COST_BPS = 1.0
TARGET_VOLATILITY = 0.10
MAX_GROSS_LEVERAGE = 4.0
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_BLOCK_DAYS = 20
RANDOM_SEED = 42


MARKETS = [
    "SP500", "NASDAQ",
    "US2", "US5", "US10", "US20",
    "CRUDE_W", "GOLD", "COPPER",
    "CORN", "WHEAT", "SOYBEAN",
    "EUR", "JPY", "GBP",
]


def find_project_root(start: Path) -> Path:
    """Locate the notebook-based futures-skew project."""
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "skew_analysis.ipynb").exists():
            return candidate
    return Path(__file__).resolve().parent if "__file__" in globals() else start


PROJECT_ROOT = find_project_root(Path.cwd())
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def locate_existing_returns_file() -> Path | None:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "futures_returns.csv",
        PROJECT_ROOT / "processed" / "futures_returns.csv",
        PROJECT_ROOT / "futures_returns.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [
        path for path in PROJECT_ROOT.rglob("futures_returns.csv")
        if ".venv" not in path.parts and "pysystemtrade" not in path.parts
    ]
    return matches[0] if matches else None


def locate_price_folders() -> tuple[Path, Path]:
    folder_pairs = [
        (PROJECT_ROOT / "data" / "raw" / "adjusted", PROJECT_ROOT / "data" / "raw" / "multiple"),
        (PROJECT_ROOT / "data" / "adjusted", PROJECT_ROOT / "data" / "multiple"),
        (PROJECT_ROOT / "raw" / "adjusted", PROJECT_ROOT / "raw" / "multiple"),
        (PROJECT_ROOT / "adjusted", PROJECT_ROOT / "multiple"),
    ]
    for adjusted, multiple in folder_pairs:
        if all((adjusted / f"{market}.csv").exists() for market in MARKETS) and all(
            (multiple / f"{market}.csv").exists() for market in MARKETS
        ):
            return adjusted, multiple

    for adjusted in PROJECT_ROOT.rglob("adjusted"):
        if ".venv" in adjusted.parts:
            continue
        sibling = adjusted.parent / "multiple"
        if all((adjusted / f"{market}.csv").exists() for market in MARKETS) and all(
            (sibling / f"{market}.csv").exists() for market in MARKETS
        ):
            return adjusted, sibling

    raise FileNotFoundError(
        "Could not find futures_returns.csv or the downloaded adjusted/multiple "
        "price folders. Run Cells 1-7 of skew_analysis.ipynb first."
    )


def read_daily_last(path: Path, value_column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["DATETIME", value_column])
    frame["DATETIME"] = pd.to_datetime(frame["DATETIME"], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["DATETIME", value_column])
    frame["date"] = frame["DATETIME"].dt.normalize()
    return frame.groupby("date")[value_column].last().sort_index()


def build_returns_from_downloaded_prices() -> tuple[Path, pd.DataFrame]:
    adjusted_folder, multiple_folder = locate_price_folders()
    series = []
    for market in MARKETS:
        adjusted = read_daily_last(adjusted_folder / f"{market}.csv", "price")
        current = read_daily_last(multiple_folder / f"{market}.csv", "PRICE")
        aligned = pd.concat(
            [adjusted.rename("adjusted"), current.rename("current_contract")], axis=1
        ).sort_index()
        aligned["current_contract"] = aligned["current_contract"].ffill()
        market_return = aligned["adjusted"].diff() / aligned["current_contract"].abs()
        trailing_volatility = market_return.ewm(span=35, min_periods=20).std().shift(1)
        market_return = market_return.mask(market_return.abs() > 12 * trailing_volatility)
        series.append(market_return.rename(market))

    reconstructed = pd.concat(series, axis=1).loc["1990-01-01":]
    output_path = PROJECT_ROOT / "data" / "processed" / "futures_returns.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(output_path, index_label="date")
    return output_path, reconstructed


RETURNS_PATH = locate_existing_returns_file()
if RETURNS_PATH is None:
    RETURNS_PATH, _ = build_returns_from_downloaded_prices()


def calculate_skew_signal(
    returns: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling_skew = returns.rolling(
        window=lookback, min_periods=int(lookback * 0.80)
    ).skew()
    cross_sectional_rank = rolling_skew.rank(axis=1, pct=True)
    raw_signal = -(cross_sectional_rank - 0.5) * 2
    smoothing_span = max(5, round(lookback / 10))
    smoothed_signal = raw_signal.ewm(
        span=smoothing_span, min_periods=smoothing_span
    ).mean()
    return rolling_skew, smoothed_signal


def calculate_positions(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    annualized_volatility = (
        returns.ewm(span=35, min_periods=20).std().shift(1) * np.sqrt(TRADING_DAYS)
    )
    raw_positions = signal / annualized_volatility
    estimated_portfolio_risk = np.sqrt(
        (raw_positions * annualized_volatility).pow(2).sum(axis=1)
    )
    positions = raw_positions.mul(
        TARGET_VOLATILITY / estimated_portfolio_risk.replace(0, np.nan), axis=0
    )
    gross_leverage = positions.abs().sum(axis=1)
    leverage_adjustment = (
        MAX_GROSS_LEVERAGE / gross_leverage.replace(0, np.nan)
    ).clip(upper=1)
    positions = positions.mul(leverage_adjustment, axis=0)
    enough_markets = signal.notna().sum(axis=1) >= 8
    return positions.where(enough_markets, 0).fillna(0)


def run_backtest(
    desired_positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    held_positions = desired_positions.shift(1).fillna(0)
    gross_return = (held_positions * returns.fillna(0)).sum(axis=1)
    turnover = held_positions.diff().abs().sum(axis=1).fillna(0)
    transaction_cost = turnover * cost_bps / 10_000
    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": gross_return - transaction_cost,
            "turnover": turnover,
        }
    )


def calculate_trend_signal(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    trailing_return = returns.rolling(
        window=lookback, min_periods=int(lookback * 0.80)
    ).sum()
    return np.sign(trailing_return)


ASSET_CLASSES = {
    "SP500": "Equity indices",
    "NASDAQ": "Equity indices",
    "US2": "Government bonds",
    "US5": "Government bonds",
    "US10": "Government bonds",
    "US20": "Government bonds",
    "CRUDE_W": "Energy",
    "GOLD": "Metals",
    "COPPER": "Metals",
    "CORN": "Agriculture",
    "WHEAT": "Agriculture",
    "SOYBEAN": "Agriculture",
    "EUR": "Currencies",
    "JPY": "Currencies",
    "GBP": "Currencies",
}

CRISIS_PERIODS = {
    "Dot-com and recession": ("2000-03-01", "2002-12-31"),
    "Global financial crisis": ("2007-07-01", "2009-06-30"),
    "Euro-area crisis": ("2010-04-01", "2012-12-31"),
    "COVID shock": ("2020-02-01", "2020-06-30"),
    "Inflation/rate shock": ("2022-01-01", "2022-12-31"),
}

print(f"Project root: {PROJECT_ROOT}")
print(f"Returns file: {RETURNS_PATH}")
print(f"Outputs:      {OUTPUTS}")


market_returns = (
    pd.read_csv(RETURNS_PATH, parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

market_returns = market_returns.loc[:, market_returns.notna().any()]
SKEW_LOOKBACKS = (126, 252)


def component_pnl(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return net contribution, held weights, turnover, and total costs."""
    held = positions.shift(1).fillna(0.0)
    gross_contribution = held * returns.fillna(0.0)
    turnover = held.diff().abs().fillna(0.0)
    cost_contribution = turnover * cost_bps / 10_000.0
    net_contribution = gross_contribution - cost_contribution
    return net_contribution, held, turnover, cost_contribution.sum(axis=1)


skew_net_components: list[pd.DataFrame] = []
skew_held_components: list[pd.DataFrame] = []
skew_turnover_components: list[pd.DataFrame] = []
skew_strategy_components: dict[str, pd.Series] = {}

for lookback in SKEW_LOOKBACKS:
    _, signal = calculate_skew_signal(market_returns, lookback)
    positions = calculate_positions(signal, market_returns)
    net_contribution, held, turnover, _ = component_pnl(
        positions, market_returns, BASELINE_COST_BPS
    )
    skew_net_components.append(net_contribution)
    skew_held_components.append(held)
    skew_turnover_components.append(turnover)
    skew_strategy_components[f"skew_{lookback}"] = net_contribution.sum(axis=1)

skew_instrument_returns = sum(skew_net_components) / len(skew_net_components)
skew_held = sum(skew_held_components) / len(skew_held_components)
skew_turnover = sum(skew_turnover_components) / len(skew_turnover_components)
skew_returns = skew_instrument_returns.sum(axis=1).rename("Skew")

trend_positions = calculate_positions(
    calculate_trend_signal(market_returns), market_returns
)
trend_result = run_backtest(trend_positions, market_returns, BASELINE_COST_BPS)
trend_returns = trend_result["net_return"].rename("Trend")
combined_returns = (0.5 * skew_returns + 0.5 * trend_returns).rename(
    "50% skew / 50% trend"
)

strategy_returns = pd.concat(
    [skew_returns, trend_returns, combined_returns], axis=1
).sort_index()

# Keep the complete daily series, including zero warm-up rows, for auditability.
strategy_returns.to_csv(OUTPUTS / "daily_strategy_returns_advanced.csv", index_label="date")
skew_instrument_returns.to_csv(
    OUTPUTS / "skew_instrument_daily_returns.csv", index_label="date"
)
skew_held.to_csv(OUTPUTS / "skew_average_held_weights.csv", index_label="date")

print(strategy_returns.tail())
print(f"\nMarkets: {market_returns.shape[1]}")
print(f"Dates:   {market_returns.index.min().date()} to {market_returns.index.max().date()}")


def active_sample(series: pd.Series) -> pd.Series:
    series = series.dropna().astype(float)
    active = series.ne(0.0)
    return series.loc[active.idxmax():] if active.any() else series


def max_drawdown_details(series: pd.Series) -> dict[str, object]:
    series = active_sample(series)
    equity = (1.0 + series).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    prior_high = equity.loc[peak]
    future = equity.loc[trough:]
    recovered = future[future >= prior_high]
    recovery = recovered.index[0] if not recovered.empty else pd.NaT
    recovery_days = (
        int((recovery - peak).days) if pd.notna(recovery) else np.nan
    )

    underwater = drawdown.lt(0)
    groups = underwater.ne(underwater.shift()).cumsum()
    longest_underwater = int(
        underwater.groupby(groups).sum().max() if underwater.any() else 0
    )
    return {
        "peak_date": peak.date(),
        "trough_date": trough.date(),
        "recovery_date": recovery.date() if pd.notna(recovery) else "Not recovered",
        "max_drawdown": float(drawdown.min()),
        "calendar_days_peak_to_recovery": recovery_days,
        "longest_underwater_trading_days": longest_underwater,
    }


def advanced_performance(series: pd.Series, name: str) -> dict[str, object]:
    series = active_sample(series)
    equity = (1.0 + series).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual_return = float(series.mean() * TRADING_DAYS)
    annual_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside = series.clip(upper=0.0)
    downside_vol = float(np.sqrt((downside.pow(2).mean()) * TRADING_DAYS))
    years = len(series) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    monthly = (1.0 + series).resample("ME").prod() - 1.0
    yearly = (1.0 + series).resample("YE").prod() - 1.0
    var_95 = float(series.quantile(0.05))
    cvar_95 = float(series.loc[series <= var_95].mean())
    return {
        "strategy": name,
        "start": series.index.min().date(),
        "end": series.index.max().date(),
        "annual_return_arithmetic": annual_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe": annual_return / annual_vol if annual_vol else np.nan,
        "sortino": annual_return / downside_vol if downside_vol else np.nan,
        "calmar": cagr / abs(float(drawdown.min())) if drawdown.min() else np.nan,
        "max_drawdown": float(drawdown.min()),
        "daily_skewness": float(stats.skew(series, bias=False)),
        "daily_excess_kurtosis": float(stats.kurtosis(series, fisher=True, bias=False)),
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "worst_day": float(series.min()),
        "worst_month": float(monthly.min()),
        "worst_year": float(yearly.min()),
        "positive_month_share": float((monthly > 0).mean()),
        "positive_year_share": float((yearly > 0).mean()),
    }


advanced_rows = [
    advanced_performance(strategy_returns[column], column)
    for column in strategy_returns.columns
]
advanced_summary = pd.DataFrame(advanced_rows)
advanced_summary.to_csv(OUTPUTS / "advanced_performance.csv", index=False)

drawdown_details = pd.DataFrame(
    [
        {"strategy": column, **max_drawdown_details(strategy_returns[column])}
        for column in strategy_returns.columns
    ]
)
drawdown_details.to_csv(OUTPUTS / "drawdown_details.csv", index=False)

equal_vol_series: dict[str, pd.Series] = {}
equal_vol_rows: list[dict[str, object]] = []
for column in strategy_returns.columns:
    sample = active_sample(strategy_returns[column])
    realised_vol = sample.std(ddof=1) * np.sqrt(TRADING_DAYS)
    scale = TARGET_VOLATILITY / realised_vol
    scaled = sample * scale
    equal_vol_series[column] = scaled
    row = advanced_performance(scaled, column)
    row["scaling_multiplier"] = scale
    row["target_volatility"] = TARGET_VOLATILITY
    row["method"] = "Ex-post full-sample normalization; diagnostic only"
    equal_vol_rows.append(row)

equal_vol_comparison = pd.DataFrame(equal_vol_rows)
equal_vol_comparison.to_csv(OUTPUTS / "equal_vol_comparison.csv", index=False)

display_columns = [
    "strategy",
    "annual_return_arithmetic",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "worst_month",
]
print(advanced_summary[display_columns].round(4).to_string(index=False))
print("\nEqual-volatility diagnostic:")
print(equal_vol_comparison[display_columns + ["scaling_multiplier"]].round(4).to_string(index=False))


SUBPERIODS = {
    "1990-1999": ("1990-01-01", "1999-12-31"),
    "2000-2009": ("2000-01-01", "2009-12-31"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
}

subperiod_rows: list[dict[str, object]] = []
for period, (start, end) in SUBPERIODS.items():
    for column in strategy_returns.columns:
        sample = strategy_returns.loc[start:end, column]
        if sample.ne(0.0).any():
            row = advanced_performance(sample, column)
            row["period"] = period
            subperiod_rows.append(row)

complete_subperiod_results = pd.DataFrame(subperiod_rows)
complete_subperiod_results.to_csv(
    OUTPUTS / "complete_subperiod_results.csv", index=False
)

crisis_rows: list[dict[str, object]] = []
for crisis, (start, end) in CRISIS_PERIODS.items():
    for column in strategy_returns.columns:
        sample = active_sample(strategy_returns.loc[start:end, column])
        if sample.empty:
            continue
        total_return = float((1.0 + sample).prod() - 1.0)
        annual_vol = float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS))
        drawdown = (1.0 + sample).cumprod()
        drawdown = drawdown / drawdown.cummax() - 1.0
        crisis_rows.append(
            {
                "crisis": crisis,
                "strategy": column,
                "start": start,
                "end": end,
                "total_return": total_return,
                "annual_volatility": annual_vol,
                "max_drawdown": float(drawdown.min()),
                "worst_day": float(sample.min()),
            }
        )

crisis_results = pd.DataFrame(crisis_rows)
crisis_results.to_csv(OUTPUTS / "crisis_results.csv", index=False)

print(
    complete_subperiod_results[
        ["period", "strategy", "annual_return_arithmetic", "sharpe", "max_drawdown"]
    ].round(4).to_string(index=False)
)
print("\nCrisis results:")
print(crisis_results.round(4).to_string(index=False))


rolling_metrics = pd.DataFrame(index=strategy_returns.index)

for years in (3, 5):
    window = years * TRADING_DAYS
    for column in strategy_returns.columns:
        rolling_mean = strategy_returns[column].rolling(window).mean() * TRADING_DAYS
        rolling_vol = (
            strategy_returns[column].rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        )
        rolling_metrics[f"{column}_sharpe_{years}y"] = rolling_mean / rolling_vol

rolling_metrics["skew_trend_correlation_3y"] = strategy_returns["Skew"].rolling(
    3 * TRADING_DAYS
).corr(strategy_returns["Trend"])

drawdown_series = pd.DataFrame(index=strategy_returns.index)
for column in strategy_returns.columns:
    sample = (1.0 + strategy_returns[column]).cumprod()
    drawdown_series[column] = sample / sample.cummax() - 1.0

rolling_metrics.to_csv(OUTPUTS / "rolling_metrics.csv", index_label="date")
drawdown_series.to_csv(OUTPUTS / "daily_drawdowns.csv", index_label="date")

print(rolling_metrics.dropna(how="all").tail().round(3))


def newey_west_mean_test(series: pd.Series, lags: int | None = None) -> dict[str, float]:
    values = active_sample(series).to_numpy(dtype=float)
    n = len(values)
    if lags is None:
        lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    demeaned = values - values.mean()
    gamma_zero = float(demeaned @ demeaned / n)
    long_run_variance = gamma_zero
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(demeaned[lag:] @ demeaned[:-lag] / n)
        long_run_variance += 2.0 * weight * gamma

    standard_error_daily = np.sqrt(max(long_run_variance, 0.0) / n)
    t_statistic = values.mean() / standard_error_daily
    p_two_sided = 2.0 * stats.t.sf(abs(t_statistic), df=n - 1)
    return {
        "hac_lags": lags,
        "newey_west_t_stat": float(t_statistic),
        "newey_west_p_two_sided": float(p_two_sided),
        "annual_mean": float(values.mean() * TRADING_DAYS),
    }


def moving_block_bootstrap(
    series: pd.Series,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    values = active_sample(series).to_numpy(dtype=float)
    n = len(values)
    if n <= block_days:
        raise ValueError("The return series is shorter than the bootstrap block.")

    rng = np.random.default_rng(seed)
    number_of_blocks = int(np.ceil(n / block_days))
    maximum_start = n - block_days
    annual_means = np.empty(repetitions)
    sharpes = np.empty(repetitions)

    for repetition in range(repetitions):
        starts = rng.integers(0, maximum_start + 1, size=number_of_blocks)
        sample = np.concatenate(
            [values[start : start + block_days] for start in starts]
        )[:n]
        annual_mean = sample.mean() * TRADING_DAYS
        annual_vol = sample.std(ddof=1) * np.sqrt(TRADING_DAYS)
        annual_means[repetition] = annual_mean
        sharpes[repetition] = annual_mean / annual_vol if annual_vol else np.nan

    return {
        "bootstrap_repetitions": repetitions,
        "bootstrap_block_days": block_days,
        "annual_return_ci_2_5": float(np.nanquantile(annual_means, 0.025)),
        "annual_return_ci_97_5": float(np.nanquantile(annual_means, 0.975)),
        "sharpe_ci_2_5": float(np.nanquantile(sharpes, 0.025)),
        "sharpe_ci_97_5": float(np.nanquantile(sharpes, 0.975)),
    }


statistical_rows = []
for column in strategy_returns.columns:
    statistical_rows.append(
        {
            "strategy": column,
            **newey_west_mean_test(strategy_returns[column]),
            **moving_block_bootstrap(strategy_returns[column]),
        }
    )

statistical_tests = pd.DataFrame(statistical_rows)
statistical_tests.to_csv(OUTPUTS / "statistical_tests.csv", index=False)
print(statistical_tests.round(4).to_string(index=False))


instrument_rows: list[dict[str, object]] = []
skew_active_index = active_sample(skew_returns).index
total_skew_annual_return = float(
    skew_instrument_returns.loc[skew_active_index].sum(axis=1).mean()
    * TRADING_DAYS
)

for instrument in skew_instrument_returns.columns:
    contribution = skew_instrument_returns.loc[skew_active_index, instrument]
    annual_contribution = float(contribution.mean() * TRADING_DAYS)
    contribution_vol = float(contribution.std(ddof=1) * np.sqrt(TRADING_DAYS))
    instrument_rows.append(
        {
            "instrument": instrument,
            "asset_class": ASSET_CLASSES.get(instrument, "Other"),
            "annual_return_contribution": annual_contribution,
            "share_of_total_annual_return": (
                annual_contribution / total_skew_annual_return
                if total_skew_annual_return
                else np.nan
            ),
            "annual_contribution_volatility": contribution_vol,
            "average_absolute_weight": float(skew_held[instrument].abs().mean()),
            "annual_turnover": float(skew_turnover[instrument].mean() * TRADING_DAYS),
        }
    )

instrument_attribution = pd.DataFrame(instrument_rows).sort_values(
    "annual_return_contribution", ascending=False
)
instrument_attribution.to_csv(OUTPUTS / "instrument_attribution.csv", index=False)

asset_class_daily = (
    skew_instrument_returns.loc[skew_active_index]
    .T.groupby(lambda instrument: ASSET_CLASSES.get(instrument, "Other"))
    .sum()
    .T
)
asset_class_rows = []
for asset_class in asset_class_daily.columns:
    contribution = asset_class_daily[asset_class]
    asset_class_rows.append(
        {
            "asset_class": asset_class,
            "annual_return_contribution": float(contribution.mean() * TRADING_DAYS),
            "annual_contribution_volatility": float(
                contribution.std(ddof=1) * np.sqrt(TRADING_DAYS)
            ),
            "share_of_total_annual_return": float(
                contribution.mean() * TRADING_DAYS / total_skew_annual_return
            ),
        }
    )

asset_class_attribution = pd.DataFrame(asset_class_rows).sort_values(
    "annual_return_contribution", ascending=False
)
asset_class_attribution.to_csv(OUTPUTS / "asset_class_attribution.csv", index=False)

long_gross_components = []
short_gross_components = []
cost_components = []
for held, turnover in zip(skew_held_components, skew_turnover_components):
    long_gross_components.append(
        (held.clip(lower=0.0) * market_returns.fillna(0.0)).sum(axis=1)
    )
    short_gross_components.append(
        (held.clip(upper=0.0) * market_returns.fillna(0.0)).sum(axis=1)
    )
    cost_components.append(turnover.sum(axis=1) * BASELINE_COST_BPS / 10_000.0)

long_short_daily = pd.DataFrame(
    {
        "long_gross": pd.concat(long_gross_components, axis=1).mean(axis=1),
        "short_gross": pd.concat(short_gross_components, axis=1).mean(axis=1),
        "trading_cost": pd.concat(cost_components, axis=1).mean(axis=1),
    }
)
long_short_daily["net_total"] = (
    long_short_daily["long_gross"]
    + long_short_daily["short_gross"]
    - long_short_daily["trading_cost"]
)
long_short_daily.to_csv(OUTPUTS / "long_short_daily.csv", index_label="date")

long_short_summary = pd.DataFrame(
    {
        "component": ["Long gross", "Short gross", "Trading costs", "Net total"],
        "annual_contribution": [
            long_short_daily.loc[skew_active_index, "long_gross"].mean()
            * TRADING_DAYS,
            long_short_daily.loc[skew_active_index, "short_gross"].mean()
            * TRADING_DAYS,
            -long_short_daily.loc[skew_active_index, "trading_cost"].mean()
            * TRADING_DAYS,
            long_short_daily.loc[skew_active_index, "net_total"].mean()
            * TRADING_DAYS,
        ],
    }
)
long_short_summary.to_csv(OUTPUTS / "long_short_attribution.csv", index=False)

print("Instrument attribution:")
print(instrument_attribution.round(4).to_string(index=False))
print("\nAsset-class attribution:")
print(asset_class_attribution.round(4).to_string(index=False))
print("\nLong/short attribution:")
print(long_short_summary.round(4).to_string(index=False))


def skew_ensemble_for_universe(
    returns: pd.DataFrame,
    cost_bps: float = BASELINE_COST_BPS,
) -> pd.Series:
    components = []
    for lookback in SKEW_LOOKBACKS:
        _, signal = calculate_skew_signal(returns, lookback)
        positions = calculate_positions(signal, returns)
        components.append(run_backtest(positions, returns, cost_bps)["net_return"])
    return pd.concat(components, axis=1).mean(axis=1)


baseline_stats = advanced_performance(skew_returns, "All markets")
leave_one_out_rows = []

for excluded_market in market_returns.columns:
    reduced_returns = market_returns.drop(columns=excluded_market)
    reduced_skew = skew_ensemble_for_universe(reduced_returns)
    stats_row = advanced_performance(reduced_skew, f"Excluding {excluded_market}")
    leave_one_out_rows.append(
        {
            "excluded_market": excluded_market,
            "annual_return": stats_row["annual_return_arithmetic"],
            "sharpe": stats_row["sharpe"],
            "max_drawdown": stats_row["max_drawdown"],
            "return_change_vs_all_markets": (
                stats_row["annual_return_arithmetic"]
                - baseline_stats["annual_return_arithmetic"]
            ),
            "sharpe_change_vs_all_markets": stats_row["sharpe"] - baseline_stats["sharpe"],
        }
    )

leave_one_out = pd.DataFrame(leave_one_out_rows).sort_values("sharpe")
leave_one_out.to_csv(OUTPUTS / "leave_one_out.csv", index=False)
print(leave_one_out.round(4).to_string(index=False))


PARAMETER_LOOKBACKS = (63, 126, 189, 252, 378, 504)
parameter_rows = []

for lookback in PARAMETER_LOOKBACKS:
    _, signal = calculate_skew_signal(market_returns, lookback)
    positions = calculate_positions(signal, market_returns)
    result = run_backtest(positions, market_returns, BASELINE_COST_BPS)
    result_stats = advanced_performance(result["net_return"], f"Skew {lookback} days")
    parameter_rows.append(
        {
            "lookback_days": lookback,
            "annual_return": result_stats["annual_return_arithmetic"],
            "annual_volatility": result_stats["annual_volatility"],
            "sharpe": result_stats["sharpe"],
            "max_drawdown": result_stats["max_drawdown"],
            "annual_turnover": float(result["turnover"].mean() * TRADING_DAYS),
            "annual_cost": float(
                result["transaction_cost"].mean() * TRADING_DAYS
            ),
        }
    )

parameter_sensitivity = pd.DataFrame(parameter_rows)
parameter_sensitivity.to_csv(OUTPUTS / "parameter_sensitivity.csv", index=False)
print(parameter_sensitivity.round(4).to_string(index=False))


sns.set_theme(style="whitegrid")

# Log-scale wealth chart
equity = (1.0 + strategy_returns).cumprod()
ax = equity.plot(figsize=(12, 6), linewidth=1.4, logy=True)
ax.set_title("Futures skew versus trend — growth of $1 on a log scale")
ax.set_xlabel("")
ax.set_ylabel("Growth of $1, logarithmic scale")
plt.tight_layout()
plt.savefig(OUTPUTS / "equity_curves_log.png", dpi=180)
plt.show()

# Rolling three-year Sharpe
rolling_sharpe_columns = [
    column for column in rolling_metrics.columns if column.endswith("sharpe_3y")
]
ax = rolling_metrics[rolling_sharpe_columns].plot(figsize=(12, 6), linewidth=1.2)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_title("Rolling three-year Sharpe ratios")
ax.set_xlabel("")
ax.set_ylabel("Sharpe ratio")
plt.tight_layout()
plt.savefig(OUTPUTS / "rolling_sharpe_3y.png", dpi=180)
plt.show()

# Rolling correlation
ax = rolling_metrics["skew_trend_correlation_3y"].plot(
    figsize=(12, 5), color="purple", linewidth=1.2
)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_title("Rolling three-year correlation: skew versus trend")
ax.set_xlabel("")
ax.set_ylabel("Correlation")
plt.tight_layout()
plt.savefig(OUTPUTS / "rolling_correlation_3y.png", dpi=180)
plt.show()

# Drawdown comparison
ax = drawdown_series.plot(figsize=(12, 6), linewidth=1.1)
ax.set_title("Strategy drawdowns")
ax.set_xlabel("")
ax.set_ylabel("Drawdown")
plt.tight_layout()
plt.savefig(OUTPUTS / "drawdown_comparison.png", dpi=180)
plt.show()

# Parameter stability
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(
    parameter_sensitivity["lookback_days"],
    parameter_sensitivity["sharpe"],
    marker="o",
)
axes[0].set_title("Sharpe by skew lookback")
axes[0].set_xlabel("Lookback days")
axes[0].set_ylabel("Sharpe")
axes[1].plot(
    parameter_sensitivity["lookback_days"],
    parameter_sensitivity["annual_turnover"],
    marker="o",
    color="darkorange",
)
axes[1].set_title("Turnover by skew lookback")
axes[1].set_xlabel("Lookback days")
axes[1].set_ylabel("Annual notional turnover")
plt.tight_layout()
plt.savefig(OUTPUTS / "parameter_stability.png", dpi=180)
plt.show()

print("Charts saved to outputs/.")


EXPECTED_OUTPUTS = [
    "advanced_performance.csv",
    "equal_vol_comparison.csv",
    "drawdown_details.csv",
    "complete_subperiod_results.csv",
    "crisis_results.csv",
    "rolling_metrics.csv",
    "daily_drawdowns.csv",
    "statistical_tests.csv",
    "instrument_attribution.csv",
    "asset_class_attribution.csv",
    "long_short_attribution.csv",
    "leave_one_out.csv",
    "parameter_sensitivity.csv",
    "equity_curves_log.png",
    "rolling_sharpe_3y.png",
    "rolling_correlation_3y.png",
    "drawdown_comparison.png",
    "parameter_stability.png",
]

missing_outputs = [name for name in EXPECTED_OUTPUTS if not (OUTPUTS / name).exists()]

if missing_outputs:
    raise RuntimeError(
        "Some advanced outputs are missing. Run every cell above first:\n"
        + "\n".join(f"- {name}" for name in missing_outputs)
    )

print("Advanced testing completed successfully.")
print(f"Created {len(EXPECTED_OUTPUTS)} dashboard-ready outputs in:")
print(OUTPUTS)
