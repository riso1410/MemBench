import pytest

from pagination.pages import total_pages


def test_partial_last_page_counts():
    assert total_pages(101, 10) == 11


def test_exact_division():
    assert total_pages(100, 10) == 10


def test_zero_items():
    assert total_pages(0, 10) == 0
