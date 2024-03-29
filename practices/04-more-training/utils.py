"""
04-more-training
author: @maximellerbach
"""

import glob
import json
import os

import cv2
import numpy as np
import re
from tensorflow.keras.utils import Sequence

def load_jsons(jsons_path):
    lines = []
    for json_path in jsons_path:
        with open(json_path, 'r') as file:
            lines += file.readlines()
    return lines

def load_image_and_get_json(img_path, table, i, res):
    """
    Load the image using img_path, load the json data using json_path,
    return both as a tuple (image, json_data)
    """
    img = cv2.imread(img_path)

    if (img_path.split("/")[-1] == table['cam/image_array']):
        return img, table
    raise ValueError("Value not find! ", i, img_path, table['cam/image_array'], res[i-1]['cam/image_array'], res[i]['cam/image_array'], res[i+1]['cam/image_array'])

class DataGenerator(Sequence):
    def __init__(
        self,
        data_directory,
        transform_funcs,
        batch_size=32,
    ):
        """Init of the class.

        Args:
            data_directory (list): list or list of list of json_paths to train on.
            batch_size (int, optional): _description_. Defaults to 64.
            transform_funcs (list): list of data augm funcs of prototype: (image, label) -> (image, label)

        Raises:
            ValueError: directory should be non-empty

        Be carefull to check that the number of images and json files match !
        You can also check that the name of the images and json files match.
        """
        self.transform_funcs = transform_funcs
        self.batch_size = batch_size

        images_path = data_directory + "/images"
        self.image_paths = sorted(glob.glob(os.path.join(images_path, "*.jpg")), key=lambda x:float(re.findall("(\d+)",x.split('/')[-1])[0]))
        self.json_paths = sorted(glob.glob(os.path.join(data_directory, "*.catalog")), key=lambda x:float(re.findall("(\d+)",x.split('/')[-1])[0]))
        manifest = glob.glob(os.path.join(data_directory, "*.json"))
        assert len(self.image_paths) > 0, "no images in directory were found"
        self.result = list(map(json.loads, load_jsons(self.json_paths)))
        self.skip = sorted(json.loads(load_jsons(manifest)[-1])["deleted_indexes"], reverse=True)
        for i in self.skip:
            del self.image_paths[i]
            del self.result[i]
        self.length = len(self.image_paths)
        self.len = self.length // self.batch_size + 1
        # just check that every img / json paths does match

        for (img_p, json_p) in zip(self.image_paths, self.json_paths):
            img_name = img_p.split(os.path.sep)[-1].split(".jpg")[0]
            json_name = json_p.split(os.path.sep)[-1].split(".catalog")[0]

            assert img_name, json_name

    def __load_next(self):
        """Prepare a batch of data for training.

        X represents the input data, and Y the expected outputs (as in Y=f(X))

        Returns:
            tuple(list, list): X and Y.
        """

        X = []
        Y = []
        Z = []
        index = []

        l = np.random.randint(0, self.length, size=self.batch_size)
        for i in l:
            img_path = self.image_paths[i]
            table = self.result[i]
            image, data = load_image_and_get_json(img_path, table, i, self.result)

            for func in self.transform_funcs:
                image, data = func(image, data)
            X.append(image)
            # X += [image]
            # Y += [data["user/angle"]]
            Y.append(data["user/angle"])
            # Z += [0.3]
            # Z += [data["user/throttle"]]
            # Z.append(data["user/throttle"])
        X = np.array(X) / 255
        Y = np.array(Y)
        # Z = np.array(Z)

        return X, Y

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.__load_next()

"""
Now some transform / data augmentation functions, 
they should only activate some of the time, for example 50% for the flip.
To do so, you can use np.random.random() and check if it is below a certain threshold.
I know this is not the most elegant/performant way, but it is easy to read and understand.

you can get creative with those !
Here are two of them, you can use many more !
"""


def flip(image: np.ndarray, data: dict):
    """Simple image flip. Also flip the label."""

    rand = np.random.random()
    if rand < 0.5: # 50%
        data["user/angle"] = data["user/angle"] * -1
        image = cv2.flip(image, 0)

    return image, data


def noise(image: np.ndarray, data: dict, mult=10):
    """Add some noise to the image."""
    
    rand = np.random.random()
    if rand < 0.1: # 10%
        noise = np.random.randint(-mult, mult, dtype=np.int8)
        image = image + noise # not perfect, here there could be some unsigned overflow 

    return image, data

def fast_clear(input_image, data: dict):
    ''' input_image:  color or grayscale image
        brightness:  -255 (all black) to +255 (all white)

        returns image of same type as input_image but with
        brightness adjusted'''
    rand = np.random.random()
    if rand < 0.1: # 10%
        cv2.convertScaleAbs(input_image, input_image, 1, -50)
    return input_image, data

def fast_darkest(input_image, data: dict):
    ''' input_image:  color or grayscale image
        brightness:  -255 (all black) to +255 (all white)

        returns image of same type as input_image but with
        brightness adjusted'''
    rand = np.random.random()
    if rand < 0.1: # 10%
        cv2.convertScaleAbs(input_image, input_image, 1, 50)
    return input_image, data

def saturation(image, data):
    rand = np.random.random()
    if rand < 0.1: # 10%
        image = image.astype(np.float32)  # Convert to float32
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = hsv_image[..., 1] * 1.5
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image, data

def blur(image, data):
    rand = np.random.random()
    if rand < 0.1: # 10%
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image, data

