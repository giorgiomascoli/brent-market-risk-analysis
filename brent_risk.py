"""Brent market-risk research: ARMA diagnostics and six one-day 99% VaR methods.

Run: python brent_risk.py
Data: Yahoo Finance continuous Brent futures ticker BZ=F, 2010-01-01 to 2026-11-20.
No investment recommendation is made. Results are illustrative research outputs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless execution (Colab, CI and local machines).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, jarque_bera, norm, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

TICKER = "BZ=F"
START_DATE = "2010-01-01"
END_DATE_EXCLUSIVE = "2026-11-21"  # Last included session: 2026-11-20 (Friday).
ALPHA = 0.01
TRAIN_FRACTION = 0.80
HISTORICAL_WINDOW = 1000
EWMA_LAMBDA = 0.94
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def get_prices(*, start: str = START_DATE, end: str = END_DATE_EXCLUSIVE) -> pd.Series:
    """Download completed daily price observations with an explicit sample cutoff."""
    import yfinance as yf  # Delayed import permits offline unit testing.

    downloaded = yf.download(
        TICKER, start=start, end=end, auto_adjust=True, progress=False
    )
    if downloaded is None or downloaded.empty:
        raise RuntimeError("No data downloaded. Check the network or Yahoo Finance availability.")
    close = downloaded["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance may return a MultiIndex frame.
        close = close[TICKER] if TICKER in close.columns else close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close.name = "Close"
    if (close <= 0).any() or len(close) < 1500:
        raise ValueError("Prices must be positive and contain sufficient history.")
    if not close.index.is_unique:
        raise ValueError("Duplicate dates detected in price history.")
    return close


def log_returns(prices: pd.Series) -> pd.Series:
    result = np.log(prices / prices.shift(1)).dropna().rename("LogReturn")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite returns encountered.")
    return result


def split_chronologically(returns: pd.Series, frac: float = TRAIN_FRACTION):
    """Preserve date order, with no shuffled information across the holdout."""
    split = int(len(returns) * frac)
    if split < HISTORICAL_WINDOW or len(returns) - split < 40:
        raise ValueError("Insufficient train/test observations for analysis.")
    return returns.iloc[:split], returns.iloc[split:]


def diagnostics(prices: pd.Series, train: pd.Series) -> dict[str, float]:
    """Tests performed on training observations only."""
    prices_train = prices.iloc[: len(train) + 1]
    lb = acorr_ljungbox(train, lags=[10, 20], return_df=True)
    lb2 = acorr_ljungbox(train**2, lags=[10, 20], return_df=True)
    return {
        "ADF prices p": float(adfuller(prices_train, autolag="AIC")[1]),
        "ADF returns p": float(adfuller(train, autolag="AIC")[1]),
        "Ljung-Box returns lag10 p": float(lb.loc[10, "lb_pvalue"]),
        "Ljung-Box returns lag20 p": float(lb.loc[20, "lb_pvalue"]),
        "Ljung-Box squared returns lag10 p": float(lb2.loc[10, "lb_pvalue"]),
        "Ljung-Box squared returns lag20 p": float(lb2.loc[20, "lb_pvalue"]),
        "ARCH-LM lag10 p": float(het_arch(train, nlags=10)[1]),
        "Jarque-Bera p": float(jarque_bera(train).pvalue),
    }


def select_arma(train: pd.Series, max_order: int = 3):
    """Minimize in-sample AIC over converged ARMA(p,q) candidates."""
    best = None
    # Supplying a numpy array avoids spurious date-frequency warnings.
    scaled = train.to_numpy() * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for p in range(max_order + 1):
            for q in range(max_order + 1):
                try:
                    fitted = ARIMA(scaled, order=(p, 0, q)).fit(
                        method_kwargs={"maxiter": 500}
                    )
                    if not fitted.mle_retvals.get("converged", True):
                        continue
                    if np.isfinite(fitted.aic) and (
                        best is None or fitted.aic < best[0]
                    ):
                        best = (float(fitted.aic), (p, q), fitted)
                except (ValueError, np.linalg.LinAlgError):
                    continue
    if best is None:
        raise RuntimeError("None of the ARMA candidates converged.")
    return best


def fit_garch(train: pd.Series, distribution: str):
    """Constant-mean GARCH(1,1), independent of the preliminary ARMA model."""
    from arch import arch_model

    return arch_model(
        train * 100.0,
        mean="Constant", vol="GARCH", p=1, q=1, dist=distribution,
    ).fit(disp="off")


def ewma_var(train: pd.Series, test: pd.Series, *, decay: float = EWMA_LAMBDA,
             alpha: float = ALPHA) -> pd.Series:
    """RiskMetrics-style exponentially weighted variance of demeaned returns.

    h_(t|t-1) = decay*h_(t-1|t-2) + (1-decay)*e_(t-1)^2.
    Forecast before observing the current test return (no look-ahead).
    """
    mu = float(train.mean())
    h = float(train.var(ddof=1))
    last_residual = float(train.iloc[-1] - mu)
    values = []
    q = norm.ppf(alpha)
    for realized in test:
        h = decay * h + (1.0 - decay) * last_residual**2
        values.append(mu + q * np.sqrt(h))
        last_residual = float(realized - mu)
    return pd.Series(values, index=test.index, name="EWMA")


def garch_var(fitted, test: pd.Series, *, alpha: float = ALPHA,
              student: bool = False) -> pd.Series:
    """One-step-ahead GARCH recursion, updating with observed test innovations."""
    params = fitted.params
    omega = float(params["omega"]) / 10000.0
    a = float(params["alpha[1]"])
    b = float(params["beta[1]"])
    mu = float(params["mu"]) / 100.0
    h = (float(fitted.conditional_volatility.iloc[-1]) / 100.0) ** 2
    last_residual = float(fitted.resid.iloc[-1]) / 100.0
    if student:
        nu = float(params["nu"])
        if nu <= 2:
            raise ValueError("Student-t GARCH must have nu > 2 for finite variance")
        # arch uses a standardized Student's t with variance 1.
        q = student_t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu)
    else:
        q = norm.ppf(alpha)
    values = []
    for realized in test:
        h = omega + a * last_residual**2 + b * h
        values.append(mu + q * np.sqrt(h))
        last_residual = float(realized - mu)
    name = "GARCH-t" if student else "GARCH-normal"
    return pd.Series(values, index=test.index, name=name)


def historical_rolling_var(returns: pd.Series, test: pd.Series, *,
                           window: int = HISTORICAL_WINDOW,
                           alpha: float = ALPHA) -> pd.Series:
    """Past window only: shift before computing each historical quantile."""
    return (
        returns.shift(1).rolling(window, min_periods=window)
        .quantile(alpha).reindex(test.index).rename("Historical-rolling")
    )


def make_forecasts(train: pd.Series, test: pd.Series,
                   fitted_normal, fitted_t) -> pd.DataFrame:
    returns = pd.concat([train, test])
    mu = float(train.mean())
    q = norm.ppf(ALPHA)
    forecasts = pd.DataFrame(index=test.index)
    forecasts["Normal-static"] = mu + q * float(train.std(ddof=1))
    forecasts["EWMA"] = ewma_var(train, test)
    forecasts["GARCH-normal"] = garch_var(fitted_normal, test)
    forecasts["GARCH-t"] = garch_var(fitted_t, test, student=True)
    forecasts["Historical-static"] = float(train.quantile(ALPHA))
    forecasts["Historical-rolling"] = historical_rolling_var(returns, test)
    if forecasts.isna().any().any():
        raise ValueError("Some VaR estimates are missing.")
    return forecasts


def _binomial_loglik(successes: int, trials: int, probability: float) -> float:
    """Stable even for 0 successes / p=0 or 1 (xlogy(0,0)=0)."""
    return float(xlogy(successes, probability) +
                 xlogy(trials - successes, 1 - probability))


def kupiec_test(breaches: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage likelihood-ratio test."""
    hits = np.asarray(breaches, dtype=bool)
    n = int(len(hits))
    if n == 0:
        raise ValueError("Empty breach series")
    x = int(hits.sum())
    ll_null = _binomial_loglik(x, n, alpha)
    ll_alternative = _binomial_loglik(x, n, x / n)
    lr = max(0.0, 2.0 * (ll_alternative - ll_null))
    return lr, float(chi2.sf(lr, df=1))


def christoffersen_independence_test(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) first-order exception independence LR test."""
    hits = np.asarray(breaches, dtype=int)
    if len(hits) < 3:
        raise ValueError("Need at least three observations")
    prev, nxt = hits[:-1], hits[1:]
    n00 = int(((prev == 0) & (nxt == 0)).sum())
    n01 = int(((prev == 0) & (nxt == 1)).sum())
    n10 = int(((prev == 1) & (nxt == 0)).sum())
    n11 = int(((prev == 1) & (nxt == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    ll_null = _binomial_loglik(n01 + n11, total, pi)
    ll_alternative = (
        _binomial_loglik(n01, n00 + n01, n01 / (n00 + n01))
        if n00 + n01 else 0.0
    ) + (
        _binomial_loglik(n11, n10 + n11, n11 / (n10 + n11))
        if n10 + n11 else 0.0
    )
    lr = max(0.0, 2.0 * (ll_alternative - ll_null))
    return lr, float(chi2.sf(lr, df=1))


def backtest(returns: pd.Series, forecasts: pd.DataFrame,
             alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exception counts, coverage, independence and tick (quantile) loss."""
    if not forecasts.index.equals(returns.index):
        raise ValueError("Forecast and return dates must match exactly")
    if forecasts.isna().any().any():
        raise ValueError("Missing forecasts cannot be backtested")
    hits = forecasts.gt(returns, axis=0)  # r_t < q_alpha,t
    rows = []
    for name in forecasts.columns:
        binary = hits[name].to_numpy()
        lr_uc, p_uc = kupiec_test(binary, alpha)
        lr_ind, p_ind = christoffersen_independence_test(binary)
        forecast = forecasts[name].to_numpy()
        actual = returns.to_numpy()
        error = actual - forecast
        pinball_loss = np.mean(np.maximum(alpha * error, (alpha - 1) * error))
        rows.append({
            "Model": name,
            "Exceptions": int(binary.sum()),
            "Exception rate (%)": 100 * float(binary.mean()),
            "Expected exceptions": len(binary) * alpha,
            "Kupiec p-value": p_uc,
            "Christoffersen independence p-value": p_ind,
            "Conditional coverage p-value": float(chi2.sf(lr_uc + lr_ind, df=2)),
            "Mean VaR threshold (%)": 100 * float(forecast.mean()),
            "Quantile loss": float(pinball_loss),
        })
    return pd.DataFrame(rows).sort_values("Quantile loss"), hits


def garch_residual_diagnostics(fitted) -> dict[str, float]:
    z = fitted.resid / fitted.conditional_volatility
    lb = acorr_ljungbox(z, lags=[10, 20], return_df=True)
    lb2 = acorr_ljungbox(z**2, lags=[10, 20], return_df=True)
    return {
        "Standardized residual LB10 p": float(lb.loc[10, "lb_pvalue"]),
        "Standardized residual LB20 p": float(lb.loc[20, "lb_pvalue"]),
        "Squared standardized residual LB10 p": float(lb2.loc[10, "lb_pvalue"]),
        "Squared standardized residual LB20 p": float(lb2.loc[20, "lb_pvalue"]),
        "Standardized residual ARCH-LM p": float(het_arch(z, nlags=10)[1]),
        "Standardized residual JB p": float(jarque_bera(z).pvalue),
        "Persistence alpha+beta": float(
            fitted.params["alpha[1]"] + fitted.params["beta[1]"]
        ),
    }


def save_figures(prices: pd.Series, returns: pd.Series, test: pd.Series,
                 forecasts: pd.DataFrame, hits: pd.DataFrame,
                 output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].plot(prices.index, prices, lw=0.85)
    ax[0].set_title("Brent continuous futures — daily closing prices")
    ax[1].plot(returns.index, 100 * returns, lw=0.6)
    ax[1].set_ylabel("Log return (%)")
    ax[1].set_title("Daily log returns")
    ax[2].plot(returns.index, 10000 * returns**2, lw=0.6)
    ax[2].set_ylabel("Squared return (pp²)")
    ax[2].set_title("Squared log returns (volatility clustering)")
    fig.tight_layout()
    fig.savefig(output_dir / "01_price_return_diagnostics.png", dpi=160)
    plt.close(fig)

    for name in forecasts.columns:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(test.index, 100 * test, lw=0.75, alpha=0.8, label="Actual return")
        ax.plot(test.index, 100 * forecasts[name], lw=1.3, label="1% VaR threshold")
        exceedance_dates = test.index[hits[name]]
        ax.scatter(exceedance_dates, 100 * test.loc[exceedance_dates],
                   marker="x", s=26, label="Exception", zorder=4)
        ax.set_title(f"{name} — out-of-sample 99% daily VaR")
        ax.set_ylabel("Daily log return (%)")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / f"var_{name.lower().replace('-', '_')}.png", dpi=160)
        plt.close(fig)


def run_analysis(prices: pd.Series, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Full workflow; pass prices explicitly for repeatable offline tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = log_returns(prices)
    train, test = split_chronologically(returns)
    preliminary = diagnostics(prices, train)
    aic, order, arma = select_arma(train)
    normal_fit = fit_garch(train, distribution="normal")
    student_fit = fit_garch(train, distribution="t")
    forecasts = make_forecasts(train, test, normal_fit, student_fit)
    summary, hits = backtest(test, forecasts)
    summary.to_csv(output_dir / "backtesting_summary.csv", index=False)
    combined = pd.concat(
        [test.rename("Actual return"), forecasts,
         hits.add_suffix(" — exception")], axis=1
    )
    combined.to_csv(output_dir / "daily_backtest.csv", index_label="Date")
    save_figures(prices, returns, test, forecasts, hits, output_dir / "figures")

    z_arma = arma.resid
    arma_lb = acorr_ljungbox(z_arma, lags=[10, 20], return_df=True)
    diagnostics_all = {
        "Sample start": str(returns.index.min().date()),
        "Sample end": str(returns.index.max().date()),
        "Train observations": len(train),
        "Test observations": len(test),
        "ARMA order (p,q)": str(order),
        "ARMA AIC": round(aic, 4),
        "ARMA resid LB10 p": float(arma_lb.loc[10, "lb_pvalue"]),
        "ARMA resid LB20 p": float(arma_lb.loc[20, "lb_pvalue"]),
        "ARMA resid ARCH-LM p": float(het_arch(z_arma, nlags=10)[1]),
        **preliminary,
        **{f"Normal-GARCH {k}": v for k, v in
           garch_residual_diagnostics(normal_fit).items()},
        **{f"Student-GARCH {k}": v for k, v in
           garch_residual_diagnostics(student_fit).items()},
        "Student-t degrees of freedom": float(student_fit.params["nu"]),
    }
    pd.Series(diagnostics_all).to_csv(output_dir / "model_diagnostics.csv",
                                      header=["Value"], index_label="Statistic")
    with (output_dir / "model_summaries.txt").open("w", encoding="utf-8") as f:
        f.write("ARMA SELECTION\n" + str(arma.summary()) + "\n\n")
        f.write("NORMAL GARCH\n" + str(normal_fit.summary()) + "\n\n")
        f.write("STUDENT-T GARCH\n" + str(student_fit.summary()) + "\n")
    print("\nRESULTS — OUT-OF-SAMPLE 99% VaR")
    print(summary.round(5).to_string(index=False))
    print(f"\nTrain: {len(train)} observations; Test: {len(test)} observations")
    print(f"Selected ARMA order: {order} (AIC={aic:.2f})")
    print(f"Generated files: {output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    prices = get_prices()
    run_analysis(prices)
