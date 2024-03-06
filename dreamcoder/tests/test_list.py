import unittest


class TestListMain(unittest.TestCase):
    def test_imports(self):
        try:
            from dreamcoder.domains.list.main import (LearnedFeatureExtractor,
                                                      isIntFunction,
                                                      isListFunction,
                                                      list_features,
                                                      list_options, main,
                                                      retrieveJSONTasks,
                                                      train_necessary)
        except Exception:
            self.fail("Unable to import list module")


if __name__ == "__main__":
    unittest.main()
