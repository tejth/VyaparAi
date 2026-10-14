"""Date helpers."""
from __future__ import annotations

import pandas as pd

_ISO = r"^\s*\d{4}-\d{1,2}-\d{1,2}"


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse to datetime; unparseable values become NaT (the caller reports them).

    Convention: ISO dates (YYYY-MM-DD) are parsed as written. Other formats such as 05/01/2026 are read
    DAY-FIRST (5 January 2026), the usual Indian convention. This is documented in the README.
    """
    s = series.astype("string").fillna("").str.strip()
    is_iso = s.str.match(_ISO, na=False).to_numpy(dtype=bool)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    rest = ~is_iso & (s.to_numpy() != "")
    if is_iso.any():
        out[is_iso] = pd.to_datetime(s[is_iso], errors="coerce", format="mixed")
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce", dayfirst=True, format="mixed")
    return out


def period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%d %b %Y} to {end:%d %b %Y}"
