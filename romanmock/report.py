"""Text layout of the reports RomanMock prints: the proposal, the catalog, its line report and the line mix.

One style everywhere, at most :data:`WIDTH` characters wide:

* a title rule, ``━━ Proposal · Hα at 1.0 ≤ z ≤ 1.2 ━━━━━━``, and section rules, ``── Potential interlopers ──────``;
* aligned ``key   value`` blocks (:func:`fields`), long values wrapped under themselves;
* tables (:func:`table`) with a rule under the header and right-aligned numbers;
* notes (:func:`notes`) wrapped under the table they explain, with footnote marks ¹ ² ³;
* progress lines, ``[romanmock] scanning 32 of 67 slices with 64 workers … done in 24 s`` (:class:`Progress`).

Tables keep the names of the line set (``Ha``, ``OIII_5007``), which are what confusion tables, interloper fractions
and ``Catalog.subset`` take; prose uses the display names (Hα, [O III] λ5007).
"""

from __future__ import annotations

import textwrap
import time
from typing import Iterable, Sequence

from .plotting import feature_label

WIDTH = 100            #: the widest line of a report
PREFIX = "[romanmock]"


def label(name: str) -> str:
    """Display name of a line (``"OIII_5007"`` -> ``"[O III] λ5007"``)."""
    return feature_label(name)


def title(text: str, width: int = WIDTH) -> str:
    head = f"━━ {text} "
    return head + "━" * max(2, width - len(head))


def section(text: str, width: int = WIDTH) -> str:
    head = f"── {text} "
    return head + "─" * max(2, width - len(head))


def notes(text: str, indent: int = 2, mark: str = "", width: int = WIDTH) -> list[str]:
    """A paragraph wrapped at ``width``; a ``mark`` (``"¹ "``) hangs in front of it."""
    return textwrap.wrap(text, width, initial_indent=" " * indent + mark, subsequent_indent=" " * (indent + len(mark)),
                         break_on_hyphens=False, break_long_words=False) or [""]


def fields(rows: Iterable[tuple[str, object]], indent: int = 2, width: int = WIDTH) -> list[str]:
    """Aligned ``key   value`` lines; rows whose value is None are left out, a long value wraps under itself."""
    rows = [(k, str(v)) for k, v in rows if v is not None]
    if not rows:
        return []
    w = max(len(k) for k, _ in rows) + 3
    out = []
    for k, v in rows:
        lines = textwrap.wrap(v, max(30, width - indent - w), break_on_hyphens=False, break_long_words=False) or [""]
        out.append(" " * indent + f"{k:<{w}s}" + lines[0])
        out += [" " * (indent + w) + line for line in lines[1:]]
    return out


def table(columns: Sequence[tuple[str, str]], rows: Sequence[Sequence], indent: int = 2, gap: int = 2) -> list[str]:
    """``columns``: ``(header, "<" or ">")``; ``rows``: one value per column (formatted with ``str``). A rule of ``─``
    under the header, as wide as each column."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, (h, _) in enumerate(columns)]
    sep = " " * gap

    def line(cells):
        return (" " * indent + sep.join(f"{c:{a}{w}}" for c, (_, a), w in zip(cells, columns, widths))).rstrip()

    return [line([h for h, _ in columns]), " " * indent + sep.join("─" * w for w in widths)] + [line(r) for r in rows]


def count(n) -> str:
    return f"{int(n):,}"


def percent(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} %"


def span(lo: float, hi: float, digits: int = 3) -> str:
    """A range, ``lo–hi``."""
    return f"{lo:.{digits}f}–{hi:.{digits}f}"


def symbols(text: str) -> str:
    """Typography for the one-line descriptions of the models: ``>=`` and ``<=`` as ≥ and ≤, ``deg2`` as deg²."""
    return text.replace(">=", "≥").replace("<=", "≤").replace("deg2", "deg²")


def zfmt(z: float) -> str:
    """A redshift as given, with at least one decimal: 1.0, 1.25, 0.33."""
    return f"{z:.1f}" if abs(10 * z - round(10 * z)) < 1e-9 else f"{z:g}"


def keep(text: str) -> str:
    """``text`` never broken across lines when wrapped (its spaces made non-breaking)."""
    return text.replace(" ", " ")


def z_cut(lo, hi, column: str = "z") -> str:
    """``1.0 ≤ z ≤ 1.2`` (either end may be None)."""
    if lo is None and hi is None:
        return f"any {column}"
    if lo is None:
        return f"{column} ≤ {zfmt(hi)}"
    if hi is None:
        return f"{column} ≥ {zfmt(lo)}"
    return f"{zfmt(lo)} ≤ {column} ≤ {zfmt(hi)}"


class Progress:
    """A progress line finished in place: ``with Progress("scanning 67 slices", verbose): ...`` prints
    ``[romanmock] scanning 67 slices … done in 24 s``."""

    def __init__(self, what: str, verbose: bool = True):
        self.what, self.verbose = what, verbose

    def __enter__(self):
        self.t0 = time.time()
        if self.verbose:
            print(f"{PREFIX} {self.what} …", end="", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.verbose:
            print(f" done in {time.time() - self.t0:.0f} s" if exc_type is None else " failed", flush=True)
        return False
