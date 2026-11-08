import pytest
import math
from coderobust.metrics.entropy import (
    calculate_shannon_entropy,
    calculate_entropy_drift,
    calculate_relative_robustness_score,
    compute_text_token_entropy
)
from coderobust.metrics.pass_evaluator import (
    execute_code_sandbox,
    extract_python_code
)

def test_shannon_entropy():
    # Uniform distribution over 2 states: H = 1.0 bit
    h = calculate_shannon_entropy([0.5, 0.5])
    assert math.isclose(h, 1.0, rel_tol=1e-3)

    # Deterministic distribution: H = 0.0
    h_zero = calculate_shannon_entropy([1.0, 0.0])
    assert math.isclose(h_zero, 0.0, rel_tol=1e-3)

def test_entropy_drift():
    drift = calculate_entropy_drift(clean_entropy=1.2, perturbed_entropy=1.8)
    assert math.isclose(drift, 0.6, rel_tol=1e-3)

def test_relative_robustness_score():
    rrs = calculate_relative_robustness_score(clean_pass_rate=80.0, perturbed_pass_rate=60.0)
    assert math.isclose(rrs, 75.0, rel_tol=1e-3)

def test_sandbox_execution_success():
    code = "def add(a, b):\n    return a + b\n"
    test = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    res = execute_code_sandbox(code, test, timeout_seconds=2.0)
    assert res["passed"] is True
    assert res["returncode"] == 0

def test_sandbox_execution_failure():
    code = "def add(a, b):\n    return a - b\n"
    test = "assert add(2, 3) == 5\n"
    res = execute_code_sandbox(code, test, timeout_seconds=2.0)
    assert res["passed"] is False

def test_sandbox_execution_timeout():
    infinite_loop = "def add(a, b):\n    while True: pass\n"
    test = "assert add(2, 3) == 5\n"
    res = execute_code_sandbox(infinite_loop, test, timeout_seconds=0.5)
    assert res["passed"] is False
    assert res["timeout"] is True

def test_compute_text_token_entropy():
    # Empty string should yield 0.0
    assert compute_text_token_entropy("") == 0.0

    # Highly repetitive token string should have low entropy
    repetitive = "x = x + x + x + x"
    h_rep = compute_text_token_entropy(repetitive)

    # Diverse token string should have higher entropy
    diverse = "def calculate_harmonic_mean(a: float, b: float, c: float) -> float:\n    return 3.0 / (1/a + 1/b + 1/c)"
    h_div = compute_text_token_entropy(diverse)

    assert h_div > h_rep
    assert h_div > 3.0
