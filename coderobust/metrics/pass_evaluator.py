import sys
import subprocess
import tempfile
import os
import re
from typing import Dict, Any, List

def extract_python_code(raw_response: str) -> str:
    """Extracts python code blocks from markdown responses."""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_response.strip()

def execute_code_sandbox(code_str: str, test_assertions: str, timeout_seconds: float = 3.0) -> Dict[str, Any]:
    """
    Executes generated code paired with test assertions inside an isolated
    safe subprocess, capturing stdout/stderr and enforcing timeouts.
    """
    full_script = f"""
from typing import List, Dict, Any, Optional, Tuple, Set
import math
import collections

{code_str}

# --- Standard Verification Assertions ---
{test_assertions}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_script)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        passed = (proc.returncode == 0)
        return {
            "passed": passed,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds}s",
            "timeout": True
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def evaluate_functional_correctness(sample_results: List[Dict[str, Any]]) -> float:
    """Computes Pass@1 as percentage of passed test instances."""
    if not sample_results:
        return 0.0
    passed_count = sum(1 for s in sample_results if s.get("passed", False))
    return (passed_count / len(sample_results)) * 100.0
