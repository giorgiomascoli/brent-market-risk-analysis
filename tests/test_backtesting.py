"""Offline tests: run with `python -m pytest -q`."""
import numpy as np
import pandas as pd
import pytest

from brent_risk import (
    backtest,
    christoffersen_independence_test,
    ewma_var,
    historical_rolling_var,
    kupiec_test,
    log_returns,
)


def test_log_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    actual = log_returns(prices)
    np.testing.assert_allclose(actual.to_numpy(), np.log([1.1, 0.9]))


def test_kupiec_exact_nominal_rate_is_not_rejected():
    hits = np.array([True] * 10 + [False] * 990)
    lr, p_value = kupiec_test(hits, alpha=0.01)
    assert lr == pytest.approx(0, abs=1e-12)
    assert p_value == pytest.approx(1)


def test_kupiec_zero_exceptions_is_finite():
    lr, p_value = kupiec_test(np.zeros(1000, dtype=bool))
    assert np.isfinite(lr) and np.isfinite(p_value)
    assert p_value < 0.01


def test_christoffersen_handles_no_exceptions():
    lr, p = christoffersen_independence_test(np.zeros(50, dtype=bool))
    assert np.isfinite(lr) and np.isfinite(p)


def test_historical_var_uses_only_past_observations():
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    r = pd.Series(np.arange(20) * 0.001, index=dates)
    test = r.iloc[10:]
    forecast = historical_rolling_var(r, test, window=5)
    assert forecast.iloc[0] == pytest.approx(r.iloc[5:10].quantile(0.01))
    changed = r.copy()
    changed.iloc[10] = -99.0
    changed_forecast = historical_rolling_var(changed, changed.iloc[10:], window=5)
    assert changed_forecast.iloc[0] == pytest.approx(forecast.iloc[0])
    assert changed_forecast.iloc[1] != pytest.approx(forecast.iloc[1])


def test_ewma_no_current_day_lookahead():
    train = pd.Series([0.01, 0.02, -0.01, 0.0, -0.02])
    test = pd.Series([0.005, -0.01, 0.008])
    initial = ewma_var(train, test)
    changed = test.copy()
    changed.iloc[0] = -0.25
    new = ewma_var(train, changed)
    assert new.iloc[0] == pytest.approx(initial.iloc[0])
    assert new.iloc[1] != pytest.approx(initial.iloc[1])


def test_backtest_counts_and_aligns():
    idx = pd.date_range('2024-01-01', periods=6, freq='D')
    actual = pd.Series([-0.04, 0.01, -0.01, 0.03, -0.02, 0.04], index=idx)
    var = pd.DataFrame({'A': [-0.03] * 6}, index=idx)
    summary, hits = backtest(actual, var)
    assert summary.loc[0, 'Exceptions'] == 1
    assert hits['A'].to_list() == [True, False, False, False, False, False]
    with pytest.raises(ValueError):
        backtest(actual.iloc[:-1], var)


def test_garch_var_no_current_day_lookahead_and_student_t_scale():
    from types import SimpleNamespace
    from brent_risk import garch_var

    fitted = SimpleNamespace(
        params=pd.Series({"omega": 0.08, "alpha[1]": 0.10,
                          "beta[1]": 0.88, "mu": 0.02, "nu": 8.0}),
        conditional_volatility=pd.Series([1.5, 1.8]),
        resid=pd.Series([0.3, -1.1]),
    )
    idx = pd.date_range("2025-01-01", periods=3)
    test = pd.Series([0.01, -0.02, 0.005], index=idx)
    out = garch_var(fitted, test, student=False)
    t_out = garch_var(fitted, test, student=True)
    changed = test.copy()
    changed.iloc[0] = -0.50
    after_change = garch_var(fitted, changed)
    assert out.iloc[0] == pytest.approx(after_change.iloc[0])
    assert out.iloc[1] != pytest.approx(after_change.iloc[1])
    assert t_out.iloc[0] < out.iloc[0]  # Student-t, df=8, 1% left tail is heavier.


def test_all_six_forecast_series_align_on_synthetic_data():
    from types import SimpleNamespace
    from brent_risk import make_forecasts

    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-01", periods=1200, freq="B")
    r = pd.Series(rng.normal(0.0001, 0.015, size=1200), index=idx)
    train, test = r.iloc[:1100], r.iloc[1100:]
    fitted = SimpleNamespace(
        params=pd.Series({"omega": 0.05, "alpha[1]": 0.1,
                          "beta[1]": 0.88, "mu": 0.01, "nu": 8.0}),
        conditional_volatility=pd.Series([1.5, 1.6]),
        resid=pd.Series([0.3, -1.1]),
    )
    forecasts = make_forecasts(train, test, fitted, fitted)
    assert forecasts.shape == (100, 6)
    assert forecasts.index.equals(test.index)
    assert np.isfinite(forecasts.to_numpy()).all()
