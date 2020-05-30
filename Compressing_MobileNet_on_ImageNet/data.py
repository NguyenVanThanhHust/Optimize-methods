import os
import sys
import time
import random
import numpy as np

from collections import defaultdict
from tqdm import tqdm

import keras
from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Model, load_model
from keras import applications
from keras import optimizers
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import Iterator
from utils import divup

class ImageListIterator(Iterator):
    """
    Iterator yielding data from a list of image names.
    
    This is based on Keras Directory Iterator but instead of reading image
    filenames from a directory, it reads them from an array.

    Arguments
    directory: Path to directory to read images from.
    """ 
    def __init__(self, directory, image_list, labels, num_class,
                image_data_generator, target_size=(256, 256),
                batch_size=32, shuffle=False, seed=None):
        self.directory = directory
        self.image_list = image_list
        self.num_samples = len(image_list)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))
        super(ImageListIterator, self).__init__(self.samples, batch_size, shuffle, seed)  

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dytpe=K.floatx())
        batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            fname = self.image_list[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 target_size=self.target_size)
            x = image.img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i, self.labels[fname]] = 1  # one-hot encoded            

        return batch_x, batch_y        

def create_new_val_sample(size=1000):
    steps = divup(size, batch_size)
    image_list = []
    for img_idx in random.sample(range(1, 500000), steps*batch__size):
        image_list.append("ILSVRC2012_val_000%05d.JPEG" % img_idx)
    return image_list

def eval_on_sample(model, val_sample):
    image_generator = ImageListIterator(val_data_dir, val_sample, val_labels, num_classes,
                                        val_datagen, target_size=(image_height, image_width), 
                                        batch_size=batch_size)
    steps = len(val_sample) // batch_size
    return model.evaluate_generator(gen, steps=steps)

def eval_full(model):
    """
    """
    image_generator = ImageListIterator(val_data_dir, val_sample, val_labels, num_classes,
                                        val_datagen, target_size=(image_height, image_width), 
                                        batch_size=batch_size)
    steps = len(val_sample) // batch_size
    return model.evaluate_generator(gen, steps=steps)


