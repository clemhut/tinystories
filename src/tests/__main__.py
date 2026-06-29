import sys
import unittest


def main() -> int:
    """Discover and run the smoke-test suite under ``src/tests``."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="src/tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
