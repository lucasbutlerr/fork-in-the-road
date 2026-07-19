"""Tests for fitr.dataset.universe. Plain file I/O, no network, no mocking
needed -- uses tmp_path for real temp files."""
from __future__ import annotations

import pytest

from fitr.dataset.universe import UniverseError, load_universe


def test_loads_simple_universe(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("AAPL\nMSFT\nGOOGL\n")
    assert load_universe(path) == ["AAPL", "MSFT", "GOOGL"]


def test_blank_lines_and_comments_skipped(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("AAPL\n\n# a comment\nMSFT\n   \n#another comment\nGOOGL\n")
    assert load_universe(path) == ["AAPL", "MSFT", "GOOGL"]


def test_tickers_uppercased(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("aapl\nMsft\n")
    assert load_universe(path) == ["AAPL", "MSFT"]


def test_duplicates_removed_preserving_first_occurrence_order(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("MSFT\nAAPL\nMSFT\nGOOGL\nAAPL\n")
    assert load_universe(path) == ["MSFT", "AAPL", "GOOGL"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(UniverseError):
        load_universe(tmp_path / "does_not_exist.txt")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("")
    with pytest.raises(UniverseError):
        load_universe(path)


def test_file_with_only_comments_raises(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("# nothing but comments\n# here\n")
    with pytest.raises(UniverseError):
        load_universe(path)


def test_line_with_whitespace_raises_with_line_number(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("AAPL\nMSFT GOOGL\n")
    with pytest.raises(UniverseError) as exc_info:
        load_universe(path)
    assert "2" in str(exc_info.value)
