import os
import sys
import time
import random
import numpy as np

from collections import defaultdict
import math

def divup(a, b):
    """
    Divides a by b and round up to the nearest integer
    """
    res = float(a)/float(b)
    res = math.ceil(res)
    return res

def plot_l1_norms(model, layer_ix):
    
