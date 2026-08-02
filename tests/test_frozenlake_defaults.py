import unittest

from scripts.prepare_frozenlake_dataset import DEFAULT_INSTRUCTION


class FrozenLakeDefaultsTest(unittest.TestCase):
    def test_instruction_matches_visible_rendering(self):
        self.assertIn("character", DEFAULT_INSTRUCTION)
        self.assertIn("treasure", DEFAULT_INSTRUCTION)
        self.assertIn("holes", DEFAULT_INSTRUCTION)
        self.assertNotIn("starts on S", DEFAULT_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
