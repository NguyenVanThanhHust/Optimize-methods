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

ilsvrc_path = "../../../ILSVRC2012/"
caffe_path = "../../../caffe_ilsvrc12/"

image_height = 224
image_width = 224
batch_size = 64
num_classes = 1000

