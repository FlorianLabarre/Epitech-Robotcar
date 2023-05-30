# just test that we can import the ref
try:
    import ref as main
except ImportError:
    import main


import unittest
import os

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


class TestDataProcessing(unittest.TestCase):
    def setUpClass():
        pass


if __name__ == '__main__':
    unittest.main()
