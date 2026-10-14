"""Indian-rupee formatting helpers (lakh / crore digit grouping)."""
from __future__ import annotations

import math
from typing import Any

RUPEE = "₹"


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True


def indian_group(n: int) -> str:
    """Group digits Indian-style: 12,34,567."""
    s = str(abs(int(n)))
    if len(s) <= 3:
        out = s
    else:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        out = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + out


def inr(value: Any, decimals: int = 0) -> str:
    """Format as full rupees, e.g. ₹12,34,567 or ₹1,250.50."""
    if _is_nan(value):
        return "n/a"
    v = round(float(value), decimals)
    neg = v < 0
    a = abs(v)
    whole = int(a)
    frac = f"{a - whole:.{decimals}f}"[1:] if decimals > 0 else ""
    text = f"{RUPEE}{indian_group(whole)}{frac}"
    return f"-{text}" if neg else text


def inr_short(value: Any) -> str:
    """Compact format: ₹1.25 Cr, ₹3.40 L, ₹12.5 K."""
    if _is_nan(value):
        return "n/a"
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}{RUPEE}{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}{RUPEE}{a / 1e5:.2f} L"
    if a >= 1e3:
        return f"{sign}{RUPEE}{a / 1e3:.1f} K"
    return f"{sign}{RUPEE}{a:.0f}"


def pct(value: Any, decimals: int = 1) -> str:
    if _is_nan(value):
        return "n/a"
    return f"{float(value):.{decimals}f}%"


def num(value: Any, decimals: int = 0) -> str:
    if _is_nan(value):
        return "n/a"
    v = float(value)
    if decimals == 0:
        return indian_group(int(round(v)))
    whole = int(abs(v))
    frac = f"{abs(v) - whole:.{decimals}f}"[1:]
    return ("-" if v < 0 else "") + indian_group(whole) + frac
