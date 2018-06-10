from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Dot, BatchNormalization, Activation
from keras import regularizers
from keras.constraints import unit_norm
from keras.layers import merge, dot  # works
from keras.models import Model
from keras.losses import mse, binary_crossentropy

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def get_simple_model(original_dim, latent_dim):

    input_img = Input(shape=(original_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_img)
    decoded = Dense(original_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, decoder


def plot_model_in_notebook(m):
    return SVG(model_to_dot(m).create(prog='dot', format='svg'))