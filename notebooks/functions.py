import csv
import glob
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import SVG
from keras import backend as K
from keras import regularizers
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.constraints import unit_norm
from keras.layers import Convolution1D
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten
from keras.layers import Input, Dense, Dropout, Lambda, Concatenate, concatenate, Dot, dot, BatchNormalization, \
    Activation
from keras.layers import merge, dot  # works
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from livelossplot import PlotLossesKeras

from ms2lda_feature_extraction import LoadMSP, LoadGNPS, LoadMGF, LoadGNPSSeperateCollisions


def make_spec_matrix(ms1, ms2, min_frag_mz=20, max_frag_mz=500, normalise=1000.0):
    doc_index = {}
    n_docs = len(ms1)
    i = 0
    for m in ms1:
        doc_index[m.name] = i
        i += 1
    n_words = max_frag_mz - min_frag_mz + 1
    n_docs = len(doc_index)
    doc_matrix = np.zeros((n_docs,n_words),np.double)
    for m in ms2:
        mz = m[0]
        mz_int = (int)(np.round(mz))
        doc_pos = doc_index[m[3].name]
        if mz_int <= max_frag_mz and mz_int >= min_frag_mz:
            mz_pos = mz_int - min_frag_mz
            doc_matrix[doc_pos,mz_pos] += m[2]
            
    s = doc_matrix.sum(axis=1)
    toremove = []
    for doc,doc_pos in doc_index.items():
        if s[doc_pos] == 0:
            toremove.append(doc)
    
    for doc in toremove:
        doc_pos = doc_index[doc]
        doc_matrix = np.delete(doc_matrix,(doc_pos),axis=0)
        del doc_index[doc]
        for d,p in doc_index.items():
            if p > doc_pos:
                doc_index[d] -= 1
    
    if normalise > 0:
        s = doc_matrix.max(axis=1)
        s = s[:,None]
        doc_matrix *= (normalise/s)

    return doc_index, doc_matrix

    
def prepare_data(input_dir, input_smiles, output_file, min_frag_mz=20, max_frag_mz=900, normalise=1.0):
    input_set = glob.glob(input_dir + '/*.ms')
    l = LoadGNPS()
    ms1, ms2, metadata = l.load_spectra(input_set)

    spectra_index, spectra_matrix = make_spec_matrix(ms1, ms2, min_frag_mz, max_frag_mz, normalise)   
    smiles = pd.read_csv(input_smiles, header=None)
    smiles = smiles.values.flatten().tolist()    

    data = {
        'vocab': np.arange(min_frag_mz, max_frag_mz+1),
        'spectra': spectra_matrix,
        'spectra_index': spectra_index,
        'metadata': metadata,
        'smiles': smiles
    }    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)   
        

def get_cdk_fingerprint_arr(input_file, smiles):
    fprints = {}
    with open('../data/smiles_sub.csv','r') as f:
        reader = csv.reader(f)
        for line in reader:
            fprints[line[0]] = [int(i) for i in line[1:]]        

    fingerprint_arr = np.zeros((len(smiles), 306+1))
    for i in range(len(smiles)):
        smile = smiles[i]
        fingerprint = fprints[smile]
        for bit in fingerprint:
            fingerprint_arr[i][bit] = 1
    return fingerprint_arr
            

def plot_spectra_and_fingerprint_mat(spectra_mat, fp_mat):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(spectra_mat, aspect='auto')
    ax1.set_title('Spectra')
    ax2.imshow(fp_mat, aspect='auto')
    ax2.set_title('Fingerprint')
    plt.tight_layout()
    plt.show()  
            

def plot_spectra_and_fingerprint(spectra_mat, fp_mat, idx):
    plt.subplot(121)
    plt.bar(np.arange(len(spectra_mat[idx])), spectra_mat[idx])
    plt.title('spectra')
    plt.subplot(122)
    plt.bar(np.arange(len(fp_mat[idx])), fp_mat[idx])
    plt.title('fingerprint')
    plt.tight_layout()
    plt.show()            
    

def count_filter(df, fingerprint_arr, spectra, lower=10, upper=3000):
    fingerprint_arr = np.copy(fingerprint_arr)
    spectra = np.copy(spectra)

    # filter on substructural counts
    keep = df[(df['c'] > lower) & (df['c'] < upper)]
    keep_idx = keep['pos'].values    
    fp_mat = fingerprint_arr[:, keep_idx]

    # keep rows that are not all zeros
    nnz_pos = ~(fp_mat==0).all(1)
    fp_mat = fp_mat[nnz_pos]
    spectra = spectra[nnz_pos]
    
    return keep, fp_mat, spectra    
    
    
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


def load_cdk_substructures(input_file, metadata_file, fingerprint_arr):
    substructures = []
    with open(input_file,'r') as f:
        for line in f:
            substructures.append(line.strip())
            
    s = fingerprint_arr.sum(axis=0)
    sub_count = []
    for i, sub in enumerate(substructures):
        sub_count.append((i, sub, s[i]))

    sub_count = sorted(sub_count,key = lambda x: x[1],reverse = True)
    count_df = pd.DataFrame(sub_count, columns=['pos', 's', 'c'])
    count_df = count_df.sort_values(by=['c'], ascending=False)
    
    cdk_sub_df = pd.read_csv(metadata_file)
    df = pd.merge(count_df, cdk_sub_df, left_on='s', right_on='Pattern').drop(['Bit position', 'Pattern'], axis=1)
    return df        
        

def basic_model(input_dim, output_dim, params):
    latent_dim = params['latent_dim']
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu', name='encoder')(input_layer)
    decoded = Dense(output_dim, activation='sigmoid', name='decoder')(encoded)        
    
    model = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model, encoder, decoder


def conv_model(input_dim, output_dim, params):
    latent_dim = params['latent_dim']
    l2_reg = regularizers.l2(params['l2_lambda']) if 'l2_lambda' in params else None
    input_layer = Input(batch_shape=(None, input_dim, 1))
    x = Conv1D(32, 6, activation='relu', padding='same', kernel_regularizer=l2_reg)(input_layer)
    # x = MaxPooling1D(2)(x)
    x = Conv1D(16, 6, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
    # x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    h = Dense(latent_dim, activation='relu')(x)
    y = Dense(output_dim, activation='sigmoid', name='decoder')(h)

    model = Model(input_layer, y)
    encoder = Model(input_layer, h)

    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model, encoder, decoder


def plot_model_in_notebook(m):
    return SVG(model_to_dot(m).create(prog='dot', format='svg'))
    

def compute_performance(y_pred_arr, y_true_arr):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for i in range(len(y_true_arr)):

        y_pred = np.round(y_pred_arr[i])
        y_true = y_true_arr[i]    
        y_pred_nnz = set(np.nonzero(y_pred)[0].tolist())
        y_true_nnz = set(np.nonzero(y_true)[0].tolist())    

        tp, fp, fn = get_confusion(y_pred_nnz, y_true_nnz)
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

    f1, precision, recall = get_prec_rec_f1(total_fn, total_fp, total_tp)
    return precision, recall, f1


def get_prec_rec_f1(total_fn, total_fp, total_tp):
    try:
        precision = total_tp / float(total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = total_tp / float(total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0
    return f1, precision, recall


def get_confusion(y_pred_nnz, y_true_nnz):
    # tp = selected and relevant
    tp = y_pred_nnz & y_true_nnz

    # fp = selected but not relevant
    fp = y_pred_nnz - y_true_nnz

    # fn = not selected but relevant
    fn = y_true_nnz - y_pred_nnz
    return tp, fp, fn


def plot_pred(y_pred_arr, y_true_arr, N=10, th=0, print_nnz=True):
    for i in range(N):
        if th == 0:            
            y_pred = np.round(y_pred_arr[i])
        else:
            y_pred = np.copy(y_pred_arr[i])
            pos = np.nonzero(y_pred < th)
            y_pred[pos] = 0
        y_true = y_true_arr[i]    

        if print_nnz:
            y_pred_nnz = set(np.nonzero(y_pred)[0].tolist())
            y_true_nnz = set(np.nonzero(y_true)[0].tolist())    
            print('y_true_nnz', y_true_nnz)
            print('y_pred_nnz', y_pred_nnz)        
        
        plt.bar(np.arange(len(y_true)), y_true)
        plt.bar(np.arange(len(y_pred)), -y_pred)
        plt.show()
        

def compute_fp_bit_performance(y_pred, y_true):
    y_pred = np.round(y_pred)
    y_pred_nnz = set(np.nonzero(y_pred)[0].tolist())
    y_true_nnz = set(np.nonzero(y_true)[0].tolist())

    tp, fp, fn = get_confusion(y_pred_nnz, y_true_nnz)
    total_tp = len(tp)
    total_fp = len(fp)
    total_fn = len(fn)

    f1, precision, recall = get_prec_rec_f1(total_fn, total_fp, total_tp)
    return total_tp, total_fp, total_fn, precision, recall, f1
    
    
def train_model(model, encoder, decoder, x_train, x_test, y_train, y_test, epochs=50, batch_size=32):
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.000001,
                            verbose=1, min_delta=1e-5)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    callbacks = [rlr]
    if is_notebook():
        callbacks.append(PlotLossesKeras())
        
    model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)
    
    y_train_pred = decoder.predict(encoder.predict(x_train))
    y_test_pred = decoder.predict(encoder.predict(x_test))
    return y_train_pred, y_test_pred

    
def eval_prec_rec(y_train, y_test, y_train_pred, y_test_pred, method):
    train_precision, train_recall, train_f1 = compute_performance(y_train_pred, y_train)
    test_precision, test_recall, test_f1 = compute_performance(y_test_pred, y_test)
    data = [
        (method, 'train', train_precision, train_recall, train_f1),
        (method, 'test', test_precision, test_recall, test_f1)
    ]
    return pd.DataFrame(data, columns=['method', 'data', 'precision', 'recall', 'f1'])

    
def eval_fp_bit_performance(y_true, y_pred, keep_df, outfile=None):
    n_spectra, n_fingerprints = y_pred.shape
    data = []
    for i in range(n_fingerprints):
        y_pred_arr = y_pred[:, i]
        y_true_arr = y_true[:, i]
        s = keep_df.iloc[i]['s']
        pos = keep_df.iloc[i]['pos']
        desc = keep_df.iloc[i]['Description']
        tp, fp, fn, prec, rec, f1 = compute_fp_bit_performance(y_pred_arr, y_true_arr)
        data.append((s, pos, desc, tp, fp, fn, prec, rec, f1))
    df = pd.DataFrame(data, columns=['Substructure', 'Bit position', 'Description', 
                                     'TP', 'FP', 'FN', 'precision', 'recall', 'f1'])                                    
    if outfile is not None:                                            
        df.to_csv(outfile, index=None, float_format='%.3f')                                     
    return df
    
    
def vis_fp_bit_performance(df, title=None):
    n_fingerprints = df.shape[0]
    plt.figure(figsize=(12,12))
    p1 = plt.bar(range(n_fingerprints), df['precision'].values, alpha=0.5)
    p2 = plt.bar(range(n_fingerprints), -df['recall'].values, alpha=0.5)
    p3 = mpatches.Patch(color='red', label='F_1')
    plt.plot(range(n_fingerprints), df['f1'].values, 'rx', markersize=10)
    if title:
        plt.title('Fingerprint bit performance (%s)' % title)
    else:
        plt.title('Fingerprint bit performance')    
    plt.legend([p1, p2, p3], ['Precision', 'Recall', 'F1'])
    plt.xticks(np.arange(n_fingerprints), df['Bit position'].values)
    ax = plt.gca()
    # ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('Fingerprint bit')
    plt.ylabel('Performance')