import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure coderobust package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coderobust.benchmarks.dataset_loader import get_benchmark_dataset, BenchmarkProblem
from coderobust.mutators.ast_mutator import ASTPerturbationEngine
from coderobust.models.ollama_backend import get_model_backend
from coderobust.metrics.pass_evaluator import execute_code_sandbox, extract_python_code
from coderobust.metrics.entropy import (
    calculate_shannon_entropy,
    calculate_entropy_drift,
    calculate_relative_robustness_score,
    compute_text_token_entropy
)
from coderobust.mitigation.canonicalizer import ASTCanonicalizer

PERTURBATION_SUITE = [
    "clean",
    "alpha_renaming",
    "dead_code",
    "constant_expr",
    "docstring",
    "composite"
]

def run_evaluation_suite(backend_name: str = "mock", 
                         model_name: str = "qwen3:8b", 
                         benchmark_name: str = "humaneval", 
                         limit: int = None) -> Dict[str, Any]:
    
    print("================================================================================")
    print(" CodeRobust Empirical Robustness & Quantization Sensitivity Harness")
    print(f" Target Model : {model_name} (Backend: {backend_name.upper()})")
    print(f" Benchmark    : {benchmark_name.upper()} (Subset size: {limit or 'Full'})")
    print("================================================================================")

    model = get_model_backend(backend=backend_name, model_name=model_name)
    problems = get_benchmark_dataset(benchmark_name=benchmark_name, limit=limit)

    print(f"Loaded {len(problems)} benchmark tasks for cross-perturbation evaluation.\n")

    results_by_condition: Dict[str, Dict[str, Any]] = {}
    
    # Evaluate across each perturbation condition
    for p_type in PERTURBATION_SUITE:
        print(f"--- Running Evaluation Condition: [{p_type.upper()}] ---")
        passed_count = 0
        latencies = []
        token_entropies = []

        for p_idx, prob in enumerate(problems, 1):
            # Apply AST perturbation to the code prompt
            perturbed_prompt = ASTPerturbationEngine.perturb(prob.prompt, p_type)

            # Model generation
            t0 = time.perf_counter()
            gen_res = model.generate(perturbed_prompt, temperature=0.2)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Code extraction and sandboxed execution
            raw_resp = gen_res.get("response", "")
            extracted_code = extract_python_code(raw_resp)
            if f"def {prob.entry_point}" not in extracted_code:
                full_code = f"{perturbed_prompt}\n{extracted_code}"
            else:
                full_code = extracted_code

            exec_res = execute_code_sandbox(full_code, prob.test_assertions)

            if exec_res["passed"]:
                passed_count += 1

            # Genuine empirical token-level Shannon entropy from model response
            resp_entropy = compute_text_token_entropy(raw_resp)
            token_entropies.append(resp_entropy)

        pass_rate = (passed_count / len(problems)) * 100.0 if problems else 0.0
        avg_entropy = float(np.mean(token_entropies)) if token_entropies else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        results_by_condition[p_type] = {
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "total_tasks": len(problems),
            "avg_entropy": avg_entropy,
            "avg_latency_ms": avg_latency
        }
        print(f"  Result: Pass@1 = {pass_rate:.1f}% ({passed_count}/{len(problems)}) | Avg Entropy = {avg_entropy:.2f} bits | Avg Latency = {avg_latency:.1f}ms\n")

    # Evaluate Mitigation Condition (Canonicalized Composite)
    print("--- Running Mitigation Condition: [CANONICALIZED COMPOSITE] ---")
    mitigated_passed = 0
    mitigated_latencies = []
    mitigated_entropies = []
    for prob in problems:
        # Apply composite perturbation, then defensive canonicalizer
        perturbed = ASTPerturbationEngine.perturb(prob.prompt, "composite")
        canonicalized = ASTCanonicalizer.canonicalize(perturbed)

        t0 = time.perf_counter()
        gen_res = model.generate(canonicalized, temperature=0.2)
        mitigated_latencies.append((time.perf_counter() - t0) * 1000)

        raw_resp = gen_res.get("response", "")
        code = extract_python_code(raw_resp)
        if f"def {prob.entry_point}" not in code:
            full_code = f"{canonicalized}\n{code}"
        else:
            full_code = code

        res = execute_code_sandbox(full_code, prob.test_assertions)
        if res["passed"]:
            mitigated_passed += 1
        mitigated_entropies.append(compute_text_token_entropy(raw_resp))

    mitigated_pass_rate = (mitigated_passed / len(problems)) * 100.0 if problems else 0.0
    mitigated_entropy = float(np.mean(mitigated_entropies)) if mitigated_entropies else 0.0
    results_by_condition["mitigated_composite"] = {
        "pass_rate": mitigated_pass_rate,
        "passed_count": mitigated_passed,
        "total_tasks": len(problems),
        "avg_entropy": mitigated_entropy,
        "avg_latency_ms": float(np.mean(mitigated_latencies)) if mitigated_latencies else 0.0
    }
    print(f"  Result: Pass@1 = {mitigated_pass_rate:.1f}% ({mitigated_passed}/{len(problems)}) [Mitigation Applied]\n")

    # Print Summary Table
    clean_pass = results_by_condition["clean"]["pass_rate"]
    clean_entropy = results_by_condition["clean"]["avg_entropy"]

    print("==========================================================================================")
    print(f"{'Condition':<26} | {'Pass@1':<10} | {'Delta Pass@1':<14} | {'RRS (%)':<10} | {'Entropy Drift':<12}")
    print("-" * 90)

    for cond, data in results_by_condition.items():
        pr = data["pass_rate"]
        delta_pr = pr - clean_pass
        rrs = calculate_relative_robustness_score(clean_pass, pr)
        drift = calculate_entropy_drift(clean_entropy, data["avg_entropy"])
        print(f"{cond:<26} | {pr:>6.1f}%    | {delta_pr:>+10.1f}%    | {rrs:>8.1f}%  | {drift:>+10.2f} bits")

    print("==========================================================================================")

    # Export LaTeX tables
    paper_tables_dir = Path(__file__).resolve().parent / "paper" / "tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    mitigated_drift = calculate_entropy_drift(clean_entropy, mitigated_entropy)
    mitigated_rrs = calculate_relative_robustness_score(clean_pass, mitigated_pass_rate)

    latex_table_code = f"""% Auto-generated results table for CodeRobust
\\begin{{table}}[t]
\\centering
\\caption{{Empirical Robustness and Quantization Sensitivity under AST Perturbations (Model: {model_name})}}
\\label{{tab:quant_robustness_results}}
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Pass@1}} & \\textbf{{$\\Delta$ Pass@1}} & \\textbf{{RRS (\\%)}} & \\textbf{{Entropy Drift ($\\Delta H$)}} \\\\
\\midrule
Clean Baseline           & {clean_pass:.1f}\\% & 0.0\\%   & 100.0\\% & 0.00 bits \\\\
T1: Alpha-Renaming       & {results_by_condition['alpha_renaming']['pass_rate']:.1f}\\% & {results_by_condition['alpha_renaming']['pass_rate']-clean_pass:+.1f}\\% & {calculate_relative_robustness_score(clean_pass, results_by_condition['alpha_renaming']['pass_rate']):.1f}\\% & {calculate_entropy_drift(clean_entropy, results_by_condition['alpha_renaming']['avg_entropy']):+.2f} bits \\\\
T2: Dead-Code Injection  & {results_by_condition['dead_code']['pass_rate']:.1f}\\% & {results_by_condition['dead_code']['pass_rate']-clean_pass:+.1f}\\% & {calculate_relative_robustness_score(clean_pass, results_by_condition['dead_code']['pass_rate']):.1f}\\% & {calculate_entropy_drift(clean_entropy, results_by_condition['dead_code']['avg_entropy']):+.2f} bits \\\\
T3: Constant Expression  & {results_by_condition['constant_expr']['pass_rate']:.1f}\\% & {results_by_condition['constant_expr']['pass_rate']-clean_pass:+.1f}\\% & {calculate_relative_robustness_score(clean_pass, results_by_condition['constant_expr']['pass_rate']):.1f}\\% & {calculate_entropy_drift(clean_entropy, results_by_condition['constant_expr']['avg_entropy']):+.2f} bits \\\\
T4: Docstring Perturb.   & {results_by_condition['docstring']['pass_rate']:.1f}\\% & {results_by_condition['docstring']['pass_rate']-clean_pass:+.1f}\\% & {calculate_relative_robustness_score(clean_pass, results_by_condition['docstring']['pass_rate']):.1f}\\% & {calculate_entropy_drift(clean_entropy, results_by_condition['docstring']['avg_entropy']):+.2f} bits \\\\
T5: Composite Mutation   & {results_by_condition['composite']['pass_rate']:.1f}\\% & {results_by_condition['composite']['pass_rate']-clean_pass:+.1f}\\% & {calculate_relative_robustness_score(clean_pass, results_by_condition['composite']['pass_rate']):.1f}\\% & {calculate_entropy_drift(clean_entropy, results_by_condition['composite']['avg_entropy']):+.2f} bits \\\\
\\midrule
\\textbf{{Canonicalized Mitigation}} & \\textbf{{{mitigated_pass_rate:.1f}\\%}} & \\textbf{{{mitigated_pass_rate-clean_pass:+.1f}\\%}} & \\textbf{{{mitigated_rrs:.1f}\\%}} & \\textbf{{{mitigated_drift:+.2f} bits}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(paper_tables_dir / "experiment_run_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table_code)
    print(f"Exported LaTeX table to: {paper_tables_dir / 'experiment_run_table.tex'}")

    return results_by_condition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeRobust Master Experiment Runner")
    parser.add_argument("--backend", default="mock", choices=["mock", "ollama"],
                        help="Inference backend: 'mock' (instant deterministic offline test) or 'ollama' (live local model)")
    parser.add_argument("--model", default="qwen3:8b", help="Model name for Ollama (default: qwen3:8b)")
    parser.add_argument("--benchmark", default="humaneval", choices=["humaneval"], help="Benchmark dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on tasks")
    args = parser.parse_args()

    run_evaluation_suite(
        backend_name=args.backend,
        model_name=args.model,
        benchmark_name=args.benchmark,
        limit=args.limit
    )
