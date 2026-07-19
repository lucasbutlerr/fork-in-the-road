"""Loads a ticker universe from a plain text file -- one ticker per line.

Deliberately minimal format, but a malformed universe file silently
producing an empty or wrong scan (a stray blank file, a typo swallowed
somewhere upstream) is a nasty failure mode to debug blind -- so this
raises loudly and specifically on the common ways it can go wrong, rather
than silently returning an empty or partial list.
"""
from __future__ import annotations

from pathlib import Path


class UniverseError(Exception):
    """Raised for a missing, empty, or malformed universe file."""


def load_universe(path: str | Path) -> list[str]:
    """Reads tickers one per line. Blank lines and lines starting with '#'
    are skipped. Tickers are uppercased and de-duplicated (first
    occurrence wins, order preserved). Raises UniverseError on a missing
    file, an empty result, or a line that doesn't look like a single
    ticker (contains whitespace, suggesting a malformed row rather than
    a stray typo worth silently tolerating)."""
    path = Path(path)
    if not path.exists():
        raise UniverseError(f"Universe file not found: {path}")

    tickers: list[str] = []
    seen: set[str] = set()
    with path.open("r") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if " " in stripped or "\t" in stripped:
                raise UniverseError(
                    f"{path}:{line_num}: '{stripped}' doesn't look like a single ticker (contains whitespace)"
                )
            ticker = stripped.upper()
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)

    if not tickers:
        raise UniverseError(f"Universe file has no tickers: {path}")

    return tickers
