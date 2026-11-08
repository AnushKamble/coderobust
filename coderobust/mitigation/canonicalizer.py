import ast
import re

class ASTCanonicalizer:
    """
    Defensive pre-inference AST canonicalizer.
    Normalizes syntactic drift, strips injected dead code,
    folds trivial constant expressions, and standardizes structure before prompt tokenization.
    """
    @classmethod
    def canonicalize(cls, code_str: str) -> str:
        try:
            tree = ast.parse(code_str)
            # 1. Strip dead If(False) blocks
            cleaned_tree = cls._strip_dead_blocks(tree)
            # 2. Fold trivial algebraic constants (e.g., (1 - 1) -> 0, (2 - 1) -> 1)
            folded_tree = cls._fold_constants(cleaned_tree)
            # 3. Fix AST locations & unparse
            ast.fix_missing_locations(folded_tree)
            canonical_code = ast.unparse(folded_tree)
            # 4. Standardize whitespace
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
