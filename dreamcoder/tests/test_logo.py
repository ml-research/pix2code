import unittest


class TestLogoMain(unittest.TestCase):
    def test_imports(self):
        try:
            from dreamcoder.domains.logo.main import (Flatten, LogoFeatureCNN,
                                                      animateSolutions,
                                                      dreamFromGrammar,
                                                      enumerateDreams,
                                                      list_options, main,
                                                      outputDreams,
                                                      visualizePrimitives)
        except Exception:
            self.fail("Unable to import logo module")


if __name__ == "__main__":
    unittest.main()
