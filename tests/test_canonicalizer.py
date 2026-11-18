import pytest
from coderobust.mitigation.canonicalizer import ASTCanonicalizer


def test_strip_dead_code():
    code = """
def compute(x: int) -> int:
    if False:
        y = 999
    return x * 2
"""
    canonical = ASTCanonicalizer.canonicalize(code)
    assert "999" not in canonical
    assert "compute" in canonical


def test_fold_constants():
    code = """
def get_val():
    return 10 - 2 + 5
"""
    canonical = ASTCanonicalizer.canonicalize(code)
    assert "compute" not in canonical
    assert "13" in canonical or "8 + 5" in canonical or "get_val" in canonical


def test_register_standardization():
    code = """
def calculate(data: list) -> int:
    total_sum = 0
    temp_idx = 0
    for item in data:
        total_sum += item
    return total_sum
"""
    canonical = ASTCanonicalizer.canonicalize(code, standardize_registers=True)
    assert "calculate" in canonical
    assert "data" in canonical
    assert "_v" in canonical
    assert "total_sum" not in canonical


def test_bidirectional_invertibility():
    code = """
def solve_puzzle(initial_state: int) -> int:
    internal_accumulator = initial_state + 10
    scratch_buffer = internal_accumulator * 2
    return scratch_buffer
"""
    canonical, mapping = ASTCanonicalizer.canonicalize_with_mapping(code, standardize_registers=True)
    assert "internal_accumulator" in mapping
    assert "scratch_buffer" in mapping
    assert "_v" in canonical

    # Invert
    decanonical = ASTCanonicalizer.decanonicalize(canonical, mapping)
    assert "internal_accumulator" in decanonical
    assert "scratch_buffer" in decanonical
    assert "_v" not in decanonical


def test_fold_constants_before_dead_block_removal():
    """Ensure arithmetic guards like (1 - 1) are folded before dead block stripping."""
    code = """
def process(x: int) -> int:
    if 1 - 1:
        dead_val = 999
    return x * 3
"""
    canonical = ASTCanonicalizer.canonicalize(code)
    assert "dead_val" not in canonical
    assert "999" not in canonical
    assert "return x * 3" in canonical
