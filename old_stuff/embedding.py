
# coding: utf-8

# # Mapping SMILES to Fragment Peaks

# Peaks are represented as integer bins. We can frame this as a multi-class classification problem?
# 
# Input: SMILES embedding
# 
# Output: integer fragment classes

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
from glob import glob
import pylab as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.metrics import categorical_accuracy
from livelossplot import PlotLossesKeras
from keras.utils.vis_utils import plot_model

import h5py

from IPython.display import display, HTML
from collections import defaultdict

from rdkit import Chem


# In[3]:


np.random.seed(seed=1)


# In[4]:


from embedding_model import MoleculeVAE


# In[5]:


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


# In[6]:


def get_input_arr(smiles_list, charset):
    
    input_arr = []
    for i in range(len(smiles_list)):
        smile = smiles_list[i]
        one_hot_encoded = to_one_hot_array(smile, charset)
        input_arr.append(one_hot_encoded)

    input_arr = np.array(input_arr)
    return input_arr


# In[7]:


def encode(model, input_array):
    x_latent = model.encoder.predict(input_array)
    return x_latent


# In[8]:


def load_data(input_file):
    with open(input_file, 'r') as f:
        data = []
        for line in f:
            tokens = line.split('\t')
            smile = tokens[0].strip()
            spectra = tokens[1].strip().split(',')
            data.append((smile, spectra, ))
        return data


# In[9]:


def load_embedding_charset(charset_file):
    with open(charset_file, 'rb') as f:
        return pickle.load(f)


# In[19]:


def write_embedding_charset(charset, charset_file):
    with open(charset_file, 'wb') as f:
        pickle.dump(charset, f, protocol=2)


# ### Load embedding

# In[13]:


data_file = 'preprocessed/pubchem_500k.h5'
charset_file = 'preprocessed/pubchem_500k_charset.p'
model_file = 'models/pubchem_500k_val_loss_0.4368_val_acc_0.9654_MoleculeVAE.h5'
LATENT_DIM = 100

charset = load_embedding_charset(charset_file)
print(charset, len(charset))

model = MoleculeVAE()
model.load(charset, model_file, latent_rep_size=LATENT_DIM)


# In[20]:


# write_embedding_charset(charset, charset_file)


# ### Load data

# In[11]:


data = load_data('data/data_large.txt')
mona_smiles = [x[0] for x in data]
mona_classes = [x[1] for x in data]
input_array = get_input_arr(mona_smiles, charset)
mona_latent = encode(model, input_array)


# In[ ]:


mlb = MultiLabelBinarizer()
X = mona_latent
y = mlb.fit_transform(mona_classes)
indices = np.arange(len(X))
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X, y, indices, test_size=0.20)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# ### Set up model

# In[ ]:


EPOCHS = 1000
BATCH_SIZE = 32
OUTPUT_SIZE = y_train.shape[1]

def get_model(model_choice):
    if model_choice == 1:
        model = Sequential()
        model.add(Dense(128, input_dim=LATENT_DIM, activation='relu'))
        model.add(Dense(OUTPUT_SIZE, activation='sigmoid'))
    elif model_choice == 2:
        model = Sequential()
        model.add(Dense(64, input_dim=LATENT_DIM, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(OUTPUT_SIZE, activation='sigmoid'))
    return model
        
model = get_model(2)
model.summary()
# plot_model(model, show_shapes=True, show_layer_names=True)


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])


# In[ ]:


history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[PlotLossesKeras()])


# In[ ]:


model.save('results/embedding_large.hdf5')
with open('results/embedding_large.history', 'wb') as f:
    pickle.dump(history.history, f)


# ### Make some predictions

# In[ ]:


def predict(model, X_new, t, mlb):
    pred = (model.predict(X_new) > t).astype(int)
    return mlb.inverse_transform(pred)

# predict on some random validation data
NUM_RANDOM = 20
idx = np.random.choice(np.arange(len(X_val)), NUM_RANDOM, replace=False)
X_new = X_val[idx]
y_new = y_val[idx]
idx_new = idx_val[idx]
smiles_new = np.array(mona_smiles)[idx_new]
t = 0.5
for smile, actual, pred in zip(smiles_new, mlb.inverse_transform(y_new), predict(model, X_new, 0.5, mlb)):
    print('SMILES', smile)
    print('Actual   ', sorted(map(int, actual)))
    print('Predicted', sorted(map(int, pred)))
    print()

