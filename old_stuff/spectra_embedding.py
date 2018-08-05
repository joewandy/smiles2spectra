
# coding: utf-8

# # Fragmentation spectra embedding

# In[ ]:


from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from livelossplot import PlotLossesKeras

from functions import *

# In[ ]:


if is_notebook():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


label = 'small'
data_path = '../data/representation_3/data_%s.txt' % label
data = load_data(data_path, num_samples=100000, remove_duplicate_spectra=True, filter_count=1)

spectra = data['spectra']
if is_notebook():
    plt.matshow(spectra)
    plt.colorbar()


# In[ ]:


rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.000001,
                        verbose=1, epsilon=1e-5)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
callbacks = [rlr, early_stop]
if is_notebook():
    callbacks.append(PlotLossesKeras())


# In[ ]:


original_dim = spectra.shape[1]
latent_dim = 100
batch_size = 32
epochs = 1000


# In[ ]:


autoencoder, encoder, decoder = get_simple_model(original_dim, latent_dim)
autoencoder.summary()
if is_notebook():
    plot_model_in_notebook(autoencoder)


# In[ ]:


pos = int(len(spectra) * 0.8)
spectra_train = spectra[0:pos, :]
spectra_test = spectra[pos:, :]


# In[ ]:


autoencoder.fit(spectra_train, spectra_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(spectra_test, spectra_test),
                callbacks=callbacks)


# In[ ]:


if is_notebook():
    spectra_encoded = encoder.predict(spectra_test)
    spectra_decoded = decoder.predict(spectra_encoded)
    for idx in range(1):
        pos = np.nonzero(spectra_test[idx])
        print(data['vocab'][pos])
        plt.bar(data['vocab'], spectra_test[idx])
        plt.bar(data['vocab'], spectra_decoded[idx])
        plt.show()


# In[ ]:


autoencoder.save('../models/spectra_autoencoder_%s.h5' % label)
encoder.save('../models/spectra_encoder_%s.h5' % label)
decoder.save('../models/spectra_decoder_%s.h5' % label)

