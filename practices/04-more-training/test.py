try:
    import ref as main
except ImportError:
    import main

import glob
import json
import os
import unittest

import cv2
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


class TestTraining(unittest.TestCase):
    def setUpClass():
        if not os.path.exists(data_path):
            os.mkdir(data_path)

            # create dummy data
            for i in range(100):
                x = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
                y = np.random.random() * 2 - 1
                cv2.imwrite(os.path.join(data_path, f'{i}.png'), x)
                with open(os.path.join(data_path, f'{i}.json'), 'w') as f:
                    json.dump({'steering': y}, f)

    def test_load_data(self):
        datagen = main.DataGenerator(data_path, [])
        x_batch, y_batch = datagen[0]
        self.assertEqual(x_batch.shape, (32, 120, 160, 3))
        self.assertEqual(y_batch.shape, (32,))

    def test_train_model(self):
        model = main.build_model()
        self.assertEqual(model.input_shape, (None, 120, 160, 3))
        self.assertEqual(model.output_shape, (None, 1))

        datagen = main.DataGenerator(data_path, [])
        main.train_model(model, datagen, epochs=1)


if __name__ == '__main__':
    unittest.main()
