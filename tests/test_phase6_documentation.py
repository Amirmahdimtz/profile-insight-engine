import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_6_embedding_semantic_themes.md"


class Phase6DocumentationTests(unittest.TestCase):
    def test_documentation_keeps_phase6_open_until_local_verification(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("git switch main", text)
        self.assertIn("python -m evaluation.embedding_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("No candidate or threshold is promoted automatically", text)


if __name__ == "__main__":
    unittest.main()
