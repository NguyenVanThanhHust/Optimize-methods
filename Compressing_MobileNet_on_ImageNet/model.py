import os
import sys
import time
import random
import numpy as np

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

def load_original():
    model = MobileNet(weights="imagenet")
    model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["categorical_accuracy", "top_k_categorical_accuracy"])
    return model

def load_and_compile(model_path):
    model = load_model(model_path, custom_objects={
                      'relu6': mobilenet.relu6,
                      'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["categorical_accuracy", "top_k_categorical_accuracy"])
    return model




