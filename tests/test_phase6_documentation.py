import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase_6_embedding_semantic_themes.md"


class Phase6DocumentationTests(unittest.TestCase):
    def test_documentation_records_verified_phase6_closure(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("COMPLETE / READY_FOR_NEXT_PHASE", text)
        self.assertNotIn("IMPLEMENTED_AWAITING_LOCAL_VERIFICATION", text)
        self.assertIn("217 tests passed with no failures or errors", text)
        self.assertIn("phase6-embedding-benchmark-v2", text)
        self.assertIn("mean Recall@5", text)
        self.assertIn("theme Macro F1 at candidate threshold", text)
        self.assertIn("are **not** frozen as production-final", text)
        self.assertIn("git switch main", text)
        self.assertIn("python -m evaluation.embedding_benchmark", text)
        self.assertIn("python -m unittest discover", text)
        self.assertIn("No candidate or threshold is promoted automatically", text)


if __name__ == "__main__":
    unittest.main()
