"""Deterministic synthetic OHLC panels used to pin the Darvas-box scanners' output."""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKETS = ("us", "indian", "japan", "korea", "european")


def panels() -> dict:
    """Name -> DataFrame (High/Low/Close). Mix of trends, flat, short, NaN/zero-bearing frames."""
    out = {}
    for seed in range(12):
        rng = np.random.RandomState(seed)
        n = 40 + 15 * seed
        close = 100 * np.exp(np.cumsum(rng.normal(0.001 * (seed % 4 - 1), 0.02, n)))
        high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
        out[f"walk{seed}"] = pd.DataFrame({"High": high, "Low": low, "Close": close})
    up = np.linspace(50, 150, 60)
    out["steady_up"] = pd.DataFrame({"High": up + 1, "Low": up - 1, "Close": up})
    dn = np.linspace(150, 50, 60)
    out["steady_down"] = pd.DataFrame({"High": dn + 1, "Low": dn - 1, "Close": dn})
    flat = np.full(30, 10.0)
    out["flat"] = pd.DataFrame({"High": flat, "Low": flat, "Close": flat})
    out["short"] = pd.DataFrame({"High": [2, 3, 2], "Low": [1, 1, 1], "Close": [1.5, 2, 1.8]})
    out["empty"] = pd.DataFrame({"High": [], "Low": [], "Close": []})
    box = pd.DataFrame({
        "High": [10, 12, 11, 10.5, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9, 13],
        "Low": [9, 9.5, 9.2, 9.4, 9.6, 9.7, 9.8, 9.9, 9.5, 9.4, 12],
        "Close": [9.5, 11, 10, 10, 10, 10, 10, 10, 9.7, 9.6, 12.5]})
    out["breakout"] = box
    brk = box.copy()
    brk.iloc[-1] = [9.6, 8.0, 8.2]
    out["breakdown"] = brk
    nz = out["walk3"].copy()
    nz.iloc[10:13, :] = np.nan
    nz.iloc[20, 0] = 0
    out["nan_zero"] = nz
    out["lowercase_cols"] = out["walk2"].rename(columns=str.lower)
    out["nse_cols"] = out["walk5"].rename(columns={"High": "CH_TRADE_HIGH_PRICE", "Low": "CH_TRADE_LOW_PRICE",
                                                   "Close": "CH_CLOSING_PRICE"})
    out["missing_close"] = out["walk1"][["High", "Low"]]
    return out
