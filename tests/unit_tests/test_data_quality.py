"""Price-panel cleaning."""

import numpy as np
import pandas as pd

from analytics.data_quality import describe_panel, find_single_day_glitches, neutralize_single_day_glitches
from data_engineering.loaders import drop_implausible_moves
from tests.unit_tests.synthetic import synthetic_prices


def _notebook_glitch_loop(prices: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, int]:
    """The original notebook's per-security loop, kept here as the reference implementation."""
    pf = prices.copy()
    r = pf.pct_change()
    n = 0
    for s in pf.columns:
        rs = r[s].dropna()
        for t in rs.index[rs.abs() > threshold]:
            nxt = rs.index[rs.index > t]
            if len(nxt) and rs.loc[nxt[0]] > -threshold:
                pf.loc[t, s] = np.nan
                n += 1
    return pf.ffill(), n


def test_glitch_detection_marks_only_reverting_prints() -> None:
    prices = synthetic_prices(n_securities=3, n_days=60)
    glitch_day = prices.index[20]
    prices.loc[glitch_day, 1001] = prices.loc[glitch_day, 1001] * 0.05  # one-day crater, recovers next day
    crash_day = prices.index[40]
    prices.loc[crash_day:, 1002] = prices.loc[crash_day:, 1002] * 0.3  # genuine permanent drop
    mask = find_single_day_glitches(prices)
    assert mask.loc[glitch_day, 1001]
    assert not mask[1002].any()
    clean, n = neutralize_single_day_glitches(prices)
    assert n == 1
    assert clean.loc[glitch_day, 1001] == prices.loc[prices.index[19], 1001]
    assert clean.loc[crash_day, 1002] == prices.loc[crash_day, 1002]


def test_vectorised_cleaning_vs_notebook_loop() -> None:
    """Both implementations neutralise every crater; the vectorised version additionally
    leaves the *recovery-day* price alone (the notebook loop also blanked it because a
    +4900% rebound is itself a >50% move)."""
    rng = np.random.default_rng(7)
    prices = synthetic_prices(n_securities=12, n_days=200, seed=9)
    glitches = []
    for j in range(6):
        d = prices.index[rng.integers(5, 190)]
        glitches.append((d, prices.columns[j]))
        prices.loc[d, prices.columns[j]] *= 0.02
    ref, n_ref = _notebook_glitch_loop(prices)
    got, n_got = neutralize_single_day_glitches(prices)
    assert n_got == 6
    assert n_ref == 12  # legacy loop over-flags the rebound day
    for d, c in glitches:
        prev = prices.index[prices.index.get_loc(d) - 1]
        nxt = prices.index[prices.index.get_loc(d) + 1]
        assert got.loc[d, c] == prices.loc[prev, c] == ref.loc[d, c]
        assert got.loc[nxt, c] == prices.loc[nxt, c]  # genuine price preserved
        assert ref.loc[nxt, c] == prices.loc[prev, c]  # legacy loop overwrote it


def test_describe_panel_keys() -> None:
    info = describe_panel(synthetic_prices(n_securities=4, n_days=10))
    assert info["dates"] == 10 and info["securities"] == 4 and info["single_day_glitches"] == 0


def test_drop_implausible_moves() -> None:
    long_df = pd.DataFrame(
        {
            "security_id": [1, 1, 1, 2, 2],
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]),
            "adj_close": [100.0, 101.0, 200.0, 50.0, 51.0],
        }
    )
    clean, dropped = drop_implausible_moves(long_df, threshold=0.25)
    assert dropped == 1
    assert len(clean) == 4
    assert not ((clean["security_id"] == 1) & (clean["adj_close"] == 200.0)).any()
