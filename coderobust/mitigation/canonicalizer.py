import ast
import re
import builtins
from typing import Set, Dict, Union

class ASTCanonicalizer:
    """
    Defensive pre-inference AST canonicalizer.
    Normalizes syntactic drift, folds trivial constant expressions (Phase 1),
    strips injected dead code (Phase 2), and standardizes
    non-interface local identifier registers (Phase 3) before prompt tokenization.
    """
    @classmethod
    def canonicalize(cls, code_str: str, standardize_registers: bool = False) -> str:
        code, _ = cls.canonicalize_with_mapping(code_str, standardize_registers=standardize_registers)
        return code

    @classmethod
    def canonicalize_with_mapping(cls, code_str: str, standardize_registers: bool = False) -> tuple[str, Dict[str, str]]:
        """
        Canonicalize code and return the forward mapping {original_identifier: standardized_register}.
        Enables bidirectional inversion post-inference.
        """
        mapping: Dict[str, str] = {}
        try:
            tree = ast.parse(code_str)
            folded_tree = cls._fold_constants(tree)
            cleaned_tree = cls._strip_dead_blocks(folded_tree)
            final_tree = cleaned_tree
            if standardize_registers:
                final_tree, mapping = cls._standardize_registers(cleaned_tree)
            ast.fix_missing_locations(final_tree)
            canonical_code = ast.unparse(final_tree)
            return re.sub(r'\n\s*\n', '\n', canonical_code).strip(), mapping
        except Exception:
            return code_str, mapping

    @classmethod
    def decanonicalize(cls, code_str: str, mapping: Dict[str, str]) -> str:
        """
        Phase 4 (Inversion): Re-maps standardized registers back to original or idiomatic
        identifiers, completely preserving developer code readability and interface hygiene.
        """
        if not mapping:
            return code_str
        try:
            tree = ast.parse(code_str)
            reverse_mapping = {v: k for k, v in mapping.items()}

            class RegisterInverter(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name) -> ast.Name:
                    if node.id in reverse_mapping:
                        return ast.copy_location(ast.Name(id=reverse_mapping[node.id], ctx=node.ctx), node)
                    return node

                def visit_Global(self, node: ast.Global) -> ast.Global:
                    node.names = [reverse_mapping.get(n, n) for n in node.names]
                    return node

                def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
                    node.names = [reverse_mapping.get(n, n) for n in node.names]
                    return node

            inverted_tree = RegisterInverter().visit(tree)
            ast.fix_missing_locations(inverted_tree)
            return ast.unparse(inverted_tree)
        except Exception:
            # Fallback to regex substitution if AST parsing fails on incomplete generation
            inverted = code_str
            for orig, std in mapping.items():
                inverted = re.sub(r'\b' + re.escape(std) + r'\b', orig, inverted)
            return inverted

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
                        elif isinstance(node.op, ast.Mult):
                            return ast.copy_location(ast.Constant(value=node.left.value * node.right.value), node)
                return node

        return ConstantFolder().visit(tree)

    @classmethod
    def _standardize_registers(cls, tree: ast.AST, prefix: str = "_v") -> tuple[ast.AST, Dict[str, str]]:
        """
        Phase 3: Identifier Register Standardization via ast.NodeTransformer.
        Remaps non-interface local variables and inner parameters to standardized
        lexical registers (_v1, _v2, ...) while strictly preserving public interface
        entrypoints, parameter keywords, type annotations, imports, and builtins.
        Returns transformed AST and mapping dictionary.
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

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                self.preserved.add(node.name)
                return self.generic_visit(node)

            def _preserve_fn_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
                self.preserved.add(node.name)
                all_args = list(node.args.args)
                if hasattr(node.args, "posonlyargs"):
                    all_args.extend(node.args.posonlyargs)
                if hasattr(node.args, "kwonlyargs"):
                    all_args.extend(node.args.kwonlyargs)
                if node.args.vararg:
                    all_args.append(node.args.vararg)
                if node.args.kwarg:
                    all_args.append(node.args.kwarg)

                for arg in all_args:
                    self.interface_params.add(arg.arg)
                    if arg.annotation:
                        for sub in ast.walk(arg.annotation):
                            self.annotation_nodes.add(id(sub))
                if node.returns:
                    for sub in ast.walk(node.returns):
                        self.annotation_nodes.add(id(sub))

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                self._preserve_fn_signature(node)
                return self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                self._preserve_fn_signature(node)
                return self.generic_visit(node)

            def visit_Global(self, node: ast.Global) -> ast.Global:
                node.names = [self.mapping.get(n, n) for n in node.names]
                return node

            def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
                node.names = [self.mapping.get(n, n) for n in node.names]
                return node

            def visit_Name(self, node: ast.Name) -> ast.Name:
                if id(node) in self.annotation_nodes:
                    return node
                if node.id in self.preserved or node.id in self.interface_params or node.id.startswith("__"):
                    return node
                if node.id not in self.mapping:
                    self.counter += 1
                    self.mapping[node.id] = f"{self.prefix}{self.counter}"
                return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)

        standardizer = RegisterStandardizer()
        transformed = standardizer.visit(tree)
        return transformed, standardizer.mapping
