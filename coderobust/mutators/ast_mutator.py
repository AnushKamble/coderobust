import ast
import re
import builtins
from typing import Dict, Any, List, Set

class AlphaRenamer(ast.NodeTransformer):
    """
    T1: Deterministically renames local variables and parameter identifiers
    to synthetic tokens (_v_1, _v_2, ...), preserving builtins, function entrypoints,
    imported symbols, and type annotations.
    """
    def __init__(self, prefix: str = "_v_"):
        self.mapping: Dict[str, str] = {}
        self.counter = 0
        self.prefix = prefix
        self.preserved_names: Set[str] = set(dir(builtins)) | {
            "self", "cls", "args", "kwargs", "True", "False", "None",
            "List", "Dict", "Tuple", "Set", "Optional", "Union", "Any",
            "Iterable", "Sequence", "Callable", "Mapping", "NamedTuple",
            "Iterator", "Generator", "Type", "Text", "Literal", "Final",
            "math", "re", "sys", "collections", "itertools"
        }
        self.entrypoint: str = ""
        self.annotation_nodes: Set[int] = set()

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            self.preserved_names.add(alias.name)
            if alias.asname:
                self.preserved_names.add(alias.asname)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        for alias in node.names:
            self.preserved_names.add(alias.name)
            if alias.asname:
                self.preserved_names.add(alias.asname)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if not self.entrypoint:
            self.entrypoint = node.name
        self.preserved_names.add(node.name)

        # Mark all AST nodes inside parameter type annotations and return type annotations as preserved
        for arg in node.args.args:
            if arg.annotation:
                for sub in ast.walk(arg.annotation):
                    self.annotation_nodes.add(id(sub))
        if node.returns:
            for sub in ast.walk(node.returns):
                self.annotation_nodes.add(id(sub))

        # Rename parameter identifiers to synthetic tokens
        for arg in node.args.args:
            if arg.arg not in self.preserved_names and not arg.arg.startswith("__"):
                if arg.arg not in self.mapping:
                    self.counter += 1
                    self.mapping[arg.arg] = f"{self.prefix}{self.counter}"
                arg.arg = self.mapping[arg.arg]

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        # Never mutate type annotations
        if id(node) in self.annotation_nodes:
            return node

        if node.id in self.preserved_names or node.id.startswith("__"):
            return node

        if node.id not in self.mapping:
            self.counter += 1
            self.mapping[node.id] = f"{self.prefix}{self.counter}"

        return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)

class DeadCodeInjector(ast.NodeTransformer):
    """
    T2: Inserts guaranteed dead-code blocks (conditional blocks with False guards)
    that alter token sequence lengths and AST depth without changing operational semantics
    or trapping indentation within unfinished prompts.
    """
    def __init__(self):
        self.injected = False

    def visit_Module(self, node: ast.Module) -> ast.Module:
        dead_stmt = ast.If(
            test=ast.Constant(value=False),
            body=[ast.Assign(
                targets=[ast.Name(id="_dead_reg_0", ctx=ast.Store())],
                value=ast.Constant(value=0)
            )],
            orelse=[]
        )
        # Insert dead statement after top-level imports/docstrings
        insert_idx = 0
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                insert_idx = i + 1
        node.body.insert(insert_idx, dead_stmt)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # If visiting standalone function not in module
        self.generic_visit(node)
        return node

class ConstantExpressionMutator(ast.NodeTransformer):
    """
    T3: Replaces primitive literal constants with mathematically equivalent expressions
    (e.g., 0 -> (1 - 1), 1 -> (2 - 1)), strictly excluding boolean types.
    """
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is int and node.value == 0:
            new_node = ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Sub(),
                right=ast.Constant(value=1)
            )
            return ast.copy_location(new_node, node)
        elif type(node.value) is int and node.value == 1:
            new_node = ast.BinOp(
                left=ast.Constant(value=2),
                op=ast.Sub(),
                right=ast.Constant(value=1)
            )
            return ast.copy_location(new_node, node)
        return node

def apply_alpha_renaming(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
        transformed = AlphaRenamer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return code_str

def apply_dead_code_injection(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
        transformed = DeadCodeInjector().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return code_str

def apply_constant_expression_mutation(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
        transformed = ConstantExpressionMutator().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return code_str

def apply_docstring_perturbation(code_str: str) -> str:
    """
    T4: Perturbs docstring text by adding formal pre-conditions and modifying stylistic tone
    while leaving functional code instructions intact.
    """
    def replace_doc(match):
        orig_doc = match.group(1)
        perturbed = f"{orig_doc.strip()}\n    Note: Execution must strictly conform to functional specifications."
        return f'"""{perturbed}"""'

    return re.sub(r'"""(.*?)"""', replace_doc, code_str, count=1, flags=re.DOTALL)

def apply_composite_mutation(code_str: str) -> str:
    """
    T5: Sequentially applies Alpha-Renaming, Dead-Code Injection, Constant Mutation,
    and Docstring Perturbation to test worst-case syntactic drift.
    """
    step1 = apply_docstring_perturbation(code_str)
    step2 = apply_dead_code_injection(step1)
    step3 = apply_constant_expression_mutation(step2)
    step4 = apply_alpha_renaming(step3)
    return step4

class ASTPerturbationEngine:
    """Unified engine to apply any registered perturbation type."""
    @staticmethod
    def perturb(code_str: str, perturbation_type: str) -> str:
        if perturbation_type == "clean":
            return code_str
        elif perturbation_type == "alpha_renaming":
            return apply_alpha_renaming(code_str)
        elif perturbation_type == "dead_code":
            return apply_dead_code_injection(code_str)
        elif perturbation_type == "constant_expr":
            return apply_constant_expression_mutation(code_str)
        elif perturbation_type == "docstring":
            return apply_docstring_perturbation(code_str)
        elif perturbation_type == "composite":
            return apply_composite_mutation(code_str)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
