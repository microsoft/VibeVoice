import pytest
from backend.config import compute_per_doc_max, compute_total_max


@pytest.mark.parametrize("iterations, expected", [
    (10, 50),
    (0, 0),
    (1, 5),
    (-1, -5),  # Negative inputs currently return negative quotas; update implementation if different behavior is desired
])
def test_compute_per_doc_max(iterations, expected):
    assert compute_per_doc_max(iterations) == expected


@pytest.mark.parametrize("iterations, expected", [
    (10, 150),
    (0, 0),
    (1, 15),
    (-1, -15),  # Negative inputs currently return negative totals; update implementation if different behavior is desired
])
def test_compute_total_max(iterations, expected):
    assert compute_total_max(iterations) == expected
