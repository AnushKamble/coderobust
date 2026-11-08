import ast
import pytest
from coderobust.mutators.ast_mutator import (
    apply_alpha_renaming,
    apply_dead_code_injection,
    apply_constant_expression_mutation,
    apply_docstring_perturbation,
    apply_composite_mutation,
    ASTPerturbationEngine
)

SAMPLE_CODE = """def calculate_total(prices, tax_rate):
    \"\"\"Calculates total with tax.\"\"\"
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total
"""

def test_alpha_renaming_preserves_syntax():
    mutated = apply_alpha_renaming(SAMPLE_CODE)
    assert mutated != SAMPLE_CODE
    # Verify AST can parse it cleanly
    tree = ast.parse(mutated)
    assert tree is not None
    assert "calculate_total" in mutated  # Function name preserved

def test_dead_code_injection():
    mutated = apply_dead_code_injection(SAMPLE_CODE)
    assert "if False:" in mutated
    tree = ast.parse(mutated)
    assert tree is not None

def test_constant_expression_mutation():
    code_with_zeros = "def check(x):\n    val = 0\n    return val + 1\n"
    mutated = apply_constant_expression_mutation(code_with_zeros)
    assert "(1 - 1)" in mutated or "(2 - 1)" in mutated
    tree = ast.parse(mutated)
    assert tree is not None

def test_docstring_perturbation():
    mutated = apply_docstring_perturbation(SAMPLE_CODE)
    assert "Execution must strictly conform" in mutated
    tree = ast.parse(mutated)
    assert tree is not None

def test_composite_mutation():
    mutated = apply_composite_mutation(SAMPLE_CODE)
    tree = ast.parse(mutated)
    assert tree is not None
    assert len(mutated) > len(SAMPLE_CODE)

def test_engine_dispatch():
    for p_type in ["clean", "alpha_renaming", "dead_code", "constant_expr", "docstring", "composite"]:
        out = ASTPerturbationEngine.perturb(SAMPLE_CODE, p_type)
        assert ast.parse(out) is not None

def test_alpha_renaming_preserves_typing_annotations():
    typed_code = "from typing import List\ndef process(nums: List[float], limit: float) -> bool:\n    return len(nums) > limit\n"
    mutated = apply_alpha_renaming(typed_code)
    assert "List[float]" in mutated
    assert "-> bool:" in mutated
    assert "_v_1" in mutated or "_v_2" in mutated
    # Must execute in Python without NameError
    namespace = {}
    exec(mutated, namespace)
    assert callable(namespace["process"])
    assert namespace["process"]([1.0, 2.0, 3.0], 2.0) is True

def test_canonicalizer_folds_and_prunes():
    from coderobust.mitigation.canonicalizer import ASTCanonicalizer
    code = "if False:\n    _dead = 0\ndef foo(x):\n    val = 1 - 1\n    return val + (2 - 1)\n"
    canonical = ASTCanonicalizer.canonicalize(code)
    assert "if False:" not in canonical
    assert "val = 0" in canonical
    assert "return val + 1" in canonical
