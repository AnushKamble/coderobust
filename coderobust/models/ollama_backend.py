import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class OllamaCodeModel:
    """
    Connects to local Ollama daemon for code generation & evaluation.
    Supports local models: qwen3:8b, qwen2.5-coder:7b.
    """
    def __init__(self, model_name: str = "qwen3:8b", endpoint: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.endpoint = endpoint

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"})

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            return {
                "response": res.get("response", ""),
                "eval_count": res.get("eval_count", 0),
                "eval_duration_ms": elapsed_ms,
                "model": self.model_name
            }

class DeterministicMockModel:
    """
    Deterministic rule-based mock model for instantaneous regression testing
    and offline test-suite verification without requiring an active GPU or Ollama server.
    """
    def __init__(self, model_name: str = "mock-qwen-7b"):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, Any]:
        # Return canned canonical implementations based on prompt keywords
        response_code = ""
        if "has_close_elements" in prompt:
            response_code = """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False"""
        elif "separate_paren_groups" in prompt:
            response_code = """def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
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
    return result"""
        elif "truncate_number" in prompt:
            response_code = """def truncate_number(number: float) -> float:
    return number % 1.0"""
        elif "below_zero" in prompt:
            response_code = """def below_zero(operations: List[int]) -> bool:
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False"""
        else:
            response_code = """def solution(*args, **kwargs):\n    return True"""

        return {
            "response": f"```python\n{response_code}\n```",
            "eval_count": 85,
            "eval_duration_ms": 1.2,
            "model": self.model_name
        }

def get_model_backend(backend: str = "mock", model_name: str = "qwen3:8b"):
    if backend == "ollama":
        client = OllamaCodeModel(model_name=model_name)
        if not client.is_available():
            print(f"[Warning] Ollama unreachable at http://localhost:11434. Falling back to DeterministicMockModel.")
            return DeterministicMockModel(model_name=model_name)
        return client
    elif backend == "mock":
        return DeterministicMockModel(model_name=model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")
