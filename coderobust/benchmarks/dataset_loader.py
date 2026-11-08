from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BenchmarkProblem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_assertions: str

# Standardized Core Evaluation Problems (HumanEval / EvalPlus standard subset)
STANDARDIZED_PROBLEMS = [
    BenchmarkProblem(
        task_id="HumanEval/0",
        prompt="""from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.2], 0.3)
    True
    \"\"\"
""",
        entry_point="has_close_elements",
        canonical_solution="""    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
""",
        test_assertions="""assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.01) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
"""
    ),
    BenchmarkProblem(
        task_id="HumanEval/1",
        prompt="""from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
""",
        entry_point="separate_paren_groups",
        canonical_solution="""    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    return result
""",
        test_assertions="""assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(())((())))') == ['(()(())((())))']
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
"""
    ),
    BenchmarkProblem(
        task_id="HumanEval/2",
        prompt="""def truncate_number(number: float) -> float:
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"
""",
        entry_point="truncate_number",
        canonical_solution="""    return number % 1.0
""",
        test_assertions="""assert truncate_number(3.5) == 0.5
assert abs(truncate_number(1.33) - 0.33) < 1e-4
assert abs(truncate_number(123.456) - 0.456) < 1e-4
"""
    ),
    BenchmarkProblem(
        task_id="HumanEval/3",
        prompt="""from typing import List

def below_zero(operations: List[int]) -> bool:
    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    \"\"\"
""",
        entry_point="below_zero",
        canonical_solution="""    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
""",
        test_assertions="""assert below_zero([]) == False
assert below_zero([1, 2, -3, 1, 2, -3]) == False
assert below_zero([1, 2, -4, 5, 6]) == True
assert below_zero([1, -1, 2, -2, 5, -5, -2]) == True
assert below_zero([1, -2, 2, -2, 5, -5, 2]) == True
"""
    )
]

def get_benchmark_dataset(benchmark_name: str = "humaneval", limit: int = None) -> List[BenchmarkProblem]:
    """Retrieves benchmark test instances."""
    data = STANDARDIZED_PROBLEMS
    if limit:
        return data[:limit]
    return data
