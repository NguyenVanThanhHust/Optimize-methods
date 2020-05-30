import os
import sys
import time
import random
import numpy as np

from collections import defaultdict
import math
import matplotlib.pyplot as plt

def count_weight(model):
    return int(np.sum([K.count_parames(p) for p in set(model.trainable_weights)]))

def divup(a, b):
    """
    Divides a by b and round up to the nearest integer
    """
    res = float(a)/float(b)
    res = math.ceil(res)
    return res

def plot_l1_norms(model, layer_ix, figure_name="net_stat.png"):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list(map(lambda x: x[1], get_l1_norms(model, layer_ix))))
    plt.xlabel("Output channel", fontsize=18)
    plt.ylabel("L1 norm", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.title(model.layers[layer_ix].name, fontsize=18)
    plt.savefig(figure_name) 

def print_savings(model):
    """
    Print how much have we compressed the model by prunning
    """
    total_params = count_weight(model)
    print("Before: %d parameters" % total_params_before)
    print("After: %d parameters" % total_params_after)
    print("Saved: %d parameters" % (total_params_before - total_params_after))
    print("Compressed to %.2f%% of original" % (100*total_params_after / total_params_before))




