import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG
from embedding_model import MoleculeVAE
from keras import regularizers
from keras.constraints import unit_norm
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Dot, BatchNormalization, Activation
from keras.layers import merge, dot  # works
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.utils.vis_utils import model_to_dot


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


def load_data(data_path, num_samples=100000, remove_duplicate_spectra=False, filter_count=0):

    seen_mz = []
    spectra = []
    smiles = []

    counter = 0
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
        items = []
        for line in lines[: min(num_samples, len(lines) - 1)]:
            if line in ['\n', '\r\n']:
                continue
            try:
                sequence_in, sequence_out = line.split('\t')

                # add spectra
                mz_int_pairs = sequence_out.split(',')
                mz_list = []
                intensity_list = []
                for pair in mz_int_pairs:
                    pair = pair[1:-1]
                    mz, intensity = pair.split(' ')
                    mz_list.append(float(mz))
                    intensity_list.append(float(intensity))
                mz_arr = np.array(mz_list)
                mz_set = set(mz_list)

                # skip duplicate m/z
                if remove_duplicate_spectra:
                    found = False
                    for x in seen_mz:
                        if mz_set == x:
                            found = True
                            break
                    if found:
                        continue
                    else:
                        seen_mz.append(mz_set)

                counter += 1
                if counter % 1000 == 0:
                    print('Loaded', counter)
                intensity_arr = np.array(intensity_list)
                spectra.append((mz_arr, intensity_arr, ))
                smiles.append(sequence_in)

            except ValueError:
                continue

    # get the initial array of unique mzs
    all_mzs = []
    for mz, _ in spectra:
        all_mzs.extend(mz)
    all_mzs = np.array(sorted(set(all_mzs)))

    word_to_idx = {k: v for v, k in enumerate(all_mzs)}
    idx_to_word = {v: k for v, k in enumerate(all_mzs)}

    # populate intensity array
    spectra_arr = []
    for mz_list, intensity_list in spectra:
        x = np.zeros(len(all_mzs))
        for mz, intensity in zip(mz_list, intensity_list):
            idx = word_to_idx[mz]
            x[idx] = intensity
        spectra_arr.append(x)
    spectra_arr = np.array(spectra_arr)
    print('Before filtering array', spectra_arr.shape, 'vocab', all_mzs.shape)

    # filter intensity array
    if filter_count > 0:

        # find columns that are used <= filter_count
        temp = np.count_nonzero(spectra_arr, axis=0)
        pos = np.where(temp <= filter_count)[0]
        print(len(pos), 'mzs to remove')

        # delete them
        spectra_arr = np.delete(spectra_arr, pos, axis=1)
        all_mzs = np.delete(all_mzs, pos, axis=0)
        print('After filtering array', spectra_arr.shape, 'vocab', all_mzs.shape)

        assert(spectra_arr.shape[1] == len(all_mzs)) # same column length
        assert(spectra_arr.shape[0] == len(np.count_nonzero(spectra_arr, axis=1))) # no empty row

        # rebuild indices after delete
        word_to_idx = {k: v for v, k in enumerate(all_mzs)}
        idx_to_word = {v: k for v, k in enumerate(all_mzs)}

    data = {
        'spectra': spectra_arr,
        'smiles': smiles,
        'mz_to_idx': word_to_idx,
        'idx_to_mz': idx_to_word,
        'vocab': all_mzs
    }
    return data


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


def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))


def one_hot_index(vec, charset):
    return map(charset.index, vec)


def to_one_hot_array(smile_str, charset, max_char=120):

    filtered = []
    for c in smile_str:
        if c in charset:
            filtered.append(c)
            if len(filtered) == max_char:
                break

    charset_list = charset.tolist()
    one_hot_encoded_fn = lambda row: list(map(lambda x: one_hot_array(x, len(charset_list)),
                                                one_hot_index(row, charset_list)))

    filtered_str = ''.join(filtered)
    filtered_str = filtered_str.ljust(max_char) # pad up to max_char
    filtered_arr = np.array(one_hot_encoded_fn(filtered_str))
    return filtered_arr


def get_input_arr(smiles_list, charset):

    input_arr = []
    for i in range(len(smiles_list)):
        smile = smiles_list[i]
        one_hot_encoded = to_one_hot_array(smile, charset)
        input_arr.append(one_hot_encoded)

    input_arr = np.array(input_arr)
    return input_arr


def encode(model, input_array):
    x_latent = model.encoder.predict(input_array)
    return x_latent


def load_embedding_charset(charset_file):
    with open(charset_file, 'rb') as f:
        return pickle.load(f)


def load_molecule_vae(model_file, charset_file, latent_dim):

    charset = load_embedding_charset(charset_file)
    print(charset, len(charset))

    model = MoleculeVAE()
    model.load(charset, model_file, latent_rep_size=latent_dim)
    return model, charset


def plot_model_in_notebook(m):
    return SVG(model_to_dot(m).create(prog='dot', format='svg'))