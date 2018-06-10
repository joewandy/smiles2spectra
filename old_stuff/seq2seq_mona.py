
# coding: utf-8

# # Seq2seq

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import random

from IPython.display import display, HTML
import pylab as plt
import numpy as np

from pipeline import *


# ### Test on MoNA data

# In[ ]:


num_samples = 10000
batch_size = 32
epochs = 500
latent_dim = 64
model_choice = 1

label = 'small'
data_path = '../data/representation_2/data_%s.txt' % label
weights_file = '../models/seq2seq_mona_small_weights.h5'
train = False


# In[ ]:


X, y = load_data(data_path, num_samples, repeat=10)
n_samples = len(X)
n_timesteps_in = np.max([len(item) for item in X])
n_timesteps_out = np.max([len(item) for item in y])
n_features_in = len(set(np.concatenate([item for item in X]).tolist()))
n_features_out = len(set(np.concatenate([item for item in y]).tolist()))
print(n_timesteps_in, n_timesteps_out, n_features_in, n_features_out)


# In[ ]:


encoded_X = encode(X)
encoded_y = encode(y)


# In[ ]:


if train:
    model, additional = get_model(model_choice, latent_dim,
                                  encoded_X.max_timesteps, encoded_y.max_timesteps,
                                  encoded_X.max_features, encoded_y.max_features)
    train_model(model, encoded_X, encoded_y, model_choice,
                latent_dim, batch_size, epochs,
                encoded_X.max_timesteps, encoded_y.max_timesteps,
                encoded_X.max_features, encoded_y.max_features)
    model.save_weights(weights_file)
else:
    model, additional = get_model(model_choice, latent_dim,
                                  encoded_X.max_timesteps, encoded_y.max_timesteps,
                                  encoded_X.max_features, encoded_y.max_features,
                                  weights_file=weights_file)


# Check on some training data

# In[ ]:


nums = [x for x in range(len(X))]
random.shuffle(nums)
X_new = []
y_new = []
for i in nums[0:20]:
    X_new.append(X[i])
    y_new.append(y[i])
X_new = np.array(X_new)
y_new = np.array(y_new)


# In[ ]:


evaluate_model(X_new, y_new, encoded_X, encoded_y, model, model_choice, additional, sep=',')

