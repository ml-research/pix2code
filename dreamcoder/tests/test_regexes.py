import unittest


class TestRegexesMain(unittest.TestCase):
    def test_imports(self):
        try:
            from dreamcoder.domains.regex.main import (
                ConstantInstantiateVisitor, LearnedFeatureExtractor,
                MyJSONFeatureExtractor, main, regex_options)
        except Exception:
            self.fail("Unable to import regexes module")


if __name__ == "__main__":
    unittest.main()
