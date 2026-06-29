import ast
from pathlib import Path
import unittest


class PackageImportTests(unittest.TestCase):
    def test_tests_package_has_module_entrypoint(self) -> None:
        entrypoint = Path("src/tests/__main__.py")
        self.assertTrue(entrypoint.exists(), "src/tests/__main__.py is missing")

        tree = ast.parse(entrypoint.read_text())
        imported_main = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.tests.tests":
                imported_main = any(alias.name == "main" for alias in node.names)

        self.assertTrue(imported_main, "src/tests/__main__.py should import main from src.tests.tests")

    def test_intra_package_imports_are_src_qualified(self) -> None:
        project_files = [
            "src/embedding_model/embedding_model.py",
            "src/pos_enc/positional_encoding.py",
            "src/transformer/utils/attention/multi_head_attention.py",
            "src/transformer/utils/attention/multi_head_diff_attention.py",
            "src/transformer/utils/decoder_block.py",
            "src/transformer/transformer.py",
            "src/data/tinystories_dataset.py",
            "src/eval/eval.py",
            "src/train/train_tinystories.py",
            "src/tests/tests.py",
        ]

        forbidden_prefixes = (
            "globals",
            "tokenizer",
            "embedding_model",
            "transformer",
            "pos_enc",
            "data",
        )

        for relative_path in project_files:
            with self.subTest(file=relative_path):
                tree = ast.parse(Path(relative_path).read_text())
                bad_imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module in forbidden_prefixes or node.module.startswith(forbidden_prefixes):
                            bad_imports.append(node.module)

                self.assertEqual(
                    bad_imports,
                    [],
                    f"{relative_path} contains bare intra-package imports: {bad_imports}",
                )


if __name__ == "__main__":
    unittest.main()
