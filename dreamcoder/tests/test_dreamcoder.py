import unittest


class TestEcModule(unittest.TestCase):
    def test_imports(self):
        try:
            from dreamcoder.dreamcoder import (ECResult, commandlineArguments,
                                               default_wake_generative,
                                               ecIterator,
                                               evaluateOnTestingTasks,
                                               sleep_recognition)
        except Exception:
            self.fail("Unable to import ec module")


if __name__ == "__main__":
    unittest.main()
