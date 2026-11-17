import ast
import re
import builtins
from typing import Set, Dict, Union

class ASTCanonicalizer:
    """
    Defensive pre-inference AST canonicalizer.
    Normalizes syntactic drift, strips injected dead code (Phase 1),
    folds trivial constant expressions (Phase 2), and standardizes
    non-interface local identifier registers (Phase 3) before prompt tokenization.
    """
    @classmethod
    def canonicalize(cls, code_str: str, standardize_registers: bool = False) -> str:
        try:
            tree = ast.parse(code_str)
            # Phase 1: Strip dead If(False) blocks
            cleaned_tree = cls._strip_dead_blocks(tree)
            # Phase 2: Fold trivial algebraic constants (e.g., (1 - 1) -> 0, (2 - 1) -> 1)
            folded_tree = cls._fold_constants(cleaned_tree)
            # Phase 3: Standardize non-interface local registers if requested
            if standardize_registers:
                folded_tree = cls._standardize_registers(folded_tree)
            # Fix AST locations & unparse
            ast.fix_missing_locations(folded_tree)
            canonical_code = ast.unparse(folded_tree)
            # Standardize whitespace
            return re.sub(r'\n\s*\n', '\n', canonical_code).strip()
        except Exception:
            return code_str

    @classmethod
    def _strip_dead_blocks(cls, tree: ast.AST) -> ast.AST:
        class DeadBlockRemover(ast.NodeTransformer):
            def visit_If(self, node: ast.If) -> ast.AST:
                # If guard is literal False or 0, eliminate the dead block, retaining orelse if present
                if isinstance(node.test, ast.Constant) and (node.test.value is False or node.test.value == 0):
                    if node.orelse:
                        return [self.visit(stmt) for stmt in node.orelse]
                    return None
                return self.generic_visit(node)

        return DeadBlockRemover().visit(tree)

    @classmethod
    def _fold_constants(cls, tree: ast.AST) -> ast.AST:
        class ConstantFolder(ast.NodeTransformer):
            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                self.generic_visit(node)
                if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                    if type(node.left.value) is int and type(node.right.value) is int:
                        if isinstance(node.op, ast.Sub):
                            return ast.copy_location(ast.Constant(value=node.left.value - node.right.value), node)
                        elif isinstance(node.op, ast.Add):
                            return ast.copy_location(ast.Constant(value=node.left.value + node.right.value), node)
                return node

        return ConstantFolder().visit(tree)

    @classmethod
    def _standardize_registers(cls, tree: ast.AST, prefix: str = "_v") -> ast.AST:
        """
        Phase 3: Identifier Register Standardization via ast.NodeTransformer.
        Remaps non-interface local variables and inner parameters to standardized
        lexical registers (_v1, _v2, ...) while strictly preserving public interface
        entrypoints, parameter keywords, type annotations, imports, and builtins.
        """
        class RegisterStandardizer(ast.NodeTransformer):
            def __init__(self):
                self.mapping: Dict[str, str] = {}
                self.counter = 0
                self.prefix = prefix
                self.preserved: Set[str] = set(dir(builtins)) | {
                    "self", "cls", "args", "kwargs", "True", "False", "None",
                    "List", "Dict", "Tuple", "Set", "Optional", "Union", "Any",
                    "Iterable", "Sequence", "Callable", "Mapping", "NamedTuple",
                    "math", "re", "sys", "collections", "itertools"
                }
                self.interface_params: Set[str] = set()
                self.annotation_nodes: Set[int] = set()

            def visit_Import(self, node: ast.Import) -> ast.Import:
                for alias in node.names:
                    self.preserved.add(alias.name)
                    if alias.asname:
                        self.preserved.add(alias.asname)
                return node

            def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
                for alias in node.names:
                    self.preserved.add(alias.name)
                    if alias.asname:
                        self.preserved.add(alias.asname)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                self.preserved.add(node.name)
                # Keep top-level signature arguments preserved to maintain API contract
                for arg in node.args.args:
                    self.interface_params.add(arg.arg)
                    if arg.annotation:
                        for sub in ast.walk(arg.annotation):
                            self.annotation_nodes.add(id(sub))
                if node.returns:
                    for sub in ast.walk(node.returns):
                        self.annotation_nodes.add(id(sub))
                return self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> ast.Name:
                if id(node) in self.annotation_nodes:
                    return node
                if node.id in self.preserved or node.id in self.interface_params or node.id.startswith("__"):
                    return node
                if node.id not in self.mapping:
                    self.counter += 1
                    self.mapping[node.id] = f"{self.prefix}{self.counter}"
                return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)

        return RegisterStandardizer().visit(tree)
