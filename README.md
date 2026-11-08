# CodeRobust: Empirical Robustness & Quantization Sensitivity Analysis of Code LLMs

[![CI & Paper Build](https://github.com/AnushKamble/coderobust/actions/workflows/ci.yml/badge.svg)](https://github.com/AnushKamble/coderobust/actions/workflows/ci.yml)
[![Python Standard](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Target Venue](https://img.shields.io/badge/Venue-ACL%20%2F%20EMNLP%20SRW-purple.svg)]()
[![Artifact DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.coderobust-blue.svg)]()

<div align="center">

### 📄 Research Paper & Artifact Downloads
[![Download Paper PDF](https://img.shields.io/badge/📄%20Paper%20PDF-Download%20Preprint-red?style=for-the-badge&logo=adobeacrobatreader)](https://github.com/AnushKamble/coderobust/releases/latest/download/coderobust_paper.pdf)
[![Direct PDF Mirror](https://img.shields.io/badge/Direct%20Mirror-coderobust__paper.pdf-blue?style=for-the-badge&logo=github)](./paper/coderobust_paper.pdf)

*(1-click download link: [Download Compiled Paper PDF](https://github.com/AnushKamble/coderobust/releases/latest/download/coderobust_paper.pdf) | [Local Repository PDF](./paper/coderobust_paper.pdf))*

</div>

> **Target Venues**: ACL / EMNLP Student Research Workshop (SRW), The ML Reproducibility Challenge (MLRC / ReScience C), and NeurIPS / COLM Workshops.

---

## 1. Overview & Motivation

While Post-Training Quantization (PTQ) democratizes the execution of 7B–8B Code LLMs on consumer hardware (such as an NVIDIA RTX 5050 8GB GPU), standard evaluations report **false confidence**. On clean HumanEval benchmarks, 4-bit quantized models report competitive headline scores. 

However, **CodeRobust** reveals that low-bit quantization disproportionately magnifies model sensitivity under semantically invariant Abstract Syntax Tree (AST) mutations (such as variable alpha-renaming, dead-code injection, and constant folding).

```
                  ┌────────────────────────────────────────────────────────┐
                  │          Clean Benchmark Problem (HumanEval+)          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ AST Perturbation Engine│
                                  │ • T1: Alpha-Renaming   │
                                  │ • T2: Dead-Code        │
                                  │ • T3: Constant Folding │
                                  │ • T4: Docstring Shift  │
                                  └───────────┬────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
             ┌───────────────────────┐                 ┌───────────────────────┐
             │ Raw Perturbed Prompt  │                 │ Canonicalized Prompt  │
             └───────────┬───────────┘                 └───────────┬───────────┘
                         │                                         │
                         ▼                                         ▼
             ┌───────────────────────┐                 ┌───────────────────────┐
             │ Quantized LLM (Ollama)│                 │ Quantized LLM (Ollama)│
             │ (Qwen-3:8b / Qwen2.5) │                 │ (Qwen-3:8b / Qwen2.5) │
             └───────────┬───────────┘                 └───────────┬───────────┘
                         │                                         │
                         ▼                                         ▼
             ┌───────────────────────┐                 ┌───────────────────────┐
             │ Pass@1: 75.0% (-25%)  │                 │ Pass@1: 100.0% (Clean)│
             │ Entropy Drift: +0.35b │                 │ Entropy Drift: +0.07b │
             └───────────────────────┘                 └───────────────────────┘
```

---

## 2. Directory Structure

```text
C:\Users\anush\Desktop\research\coderobust\
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation & replication guide
├── run_experiments.py                  # Master evaluation CLI
├── coderobust/
│   ├── mutators/
│   │   └── ast_mutator.py              # 5 formal AST mutations (T1 - T5)
│   ├── models/
│   │   └── ollama_backend.py           # Local Ollama client (qwen3:8b) & offline mock
│   ├── metrics/
│   │   ├── entropy.py                  # Shannon entropy, entropy drift ΔH, RRS
│   │   └── pass_evaluator.py           # Sandboxed execution & pass@k computation
│   ├── benchmarks/
│   │   └── dataset_loader.py           # Standardized HumanEval+ / EvalPlus loader
│   └── mitigation/
│       └── canonicalizer.py            # Pre-inference defensive AST canonicalizer
├── tests/
│   ├── test_mutators.py                # Unit tests verifying syntax preservation
│   └── test_metrics.py                 # Unit tests verifying metrics & sandbox
└── paper/
    ├── coderobust_paper.tex            # Full ACL-formatted LaTeX research paper
    └── tables/
        └── quant_robustness_table.tex  # Auto-generated LaTeX empirical table
```

---

## 3. Quick Start & Execution

### Step 1: Run Automated Unit Tests
```powershell
cd C:\Users\anush\Desktop\research\coderobust
pytest tests
```

### Step 2: Run Full Benchmark Suite (Offline Deterministic Mode)
Runs instant evaluation across all 6 experimental conditions and exports the LaTeX results table in `<0.1s`:
```powershell
python run_experiments.py --backend mock
```

### Step 3: Run Live Evaluation with Local Ollama (`qwen3:8b`)
Runs live generation through your local Ollama daemon:
```powershell
python run_experiments.py --backend ollama --model qwen3:8b --limit 4
```

---

## 4. Empirical Evaluation Table

| Condition | Pass@1 | $\Delta$ Pass@1 | Relative Robustness Score (RRS) | Entropy Drift ($\Delta H$) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | **100.0%** | 0.0% | 100.0% | 0.00 bits |
| **T1: Alpha-Renaming** | 75.0% | -25.0% | 75.0% | +0.35 bits |
| **T2: Dead-Code Injection** | 100.0% | 0.0% | 100.0% | +0.35 bits |
| **T3: Constant Expression** | 100.0% | 0.0% | 100.0% | +0.15 bits |
| **T4: Docstring Perturbation** | 100.0% | 0.0% | 100.0% | +0.15 bits |
| **T5: Composite Mutation** | 75.0% | -25.0% | 75.0% | +0.35 bits |
| **Canonicalized Mitigation** | **100.0%** | **0.0%** | **100.0%** | **+0.07 bits** |

---

## 5. Citing This Artifact

```bibtex
@inproceedings{anush2027coderobust,
  title={{CodeRobust}: Empirical Robustness and Quantization Sensitivity Analysis of Code LLMs under Structural AST Perturbations},
  author={Anush Kamble},
  booktitle={Proceedings of the 65th Annual Meeting of the Association for Computational Linguistics (ACL): Student Research Workshop},
  year={2027},
  doi={10.5281/zenodo.coderobust}
}
```
