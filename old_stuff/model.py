from __future__ import print_function

import pickle
from collections import Counter

import numpy as np
import pylab as plt
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, LSTM, Dense, GRU, Masking
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from livelossplot import PlotLossesKeras


# Mostly copy and pasted from https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

def load_data(data_path, num_samples, filter_unknown=1.0):

    c = Counter()
    sequences = []
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            if line in ['\n', '\r\n']:
                continue
            input_text, target_text = line.split('\t')
            target_text = target_text.split(',')
            sequences.append((input_text, target_text))
            c.update(target_text)

    words, counts = zip(*c.most_common())
    if is_notebook():
        plt.plot(range(len(counts)), counts)
        plt.title('Target text distribution')
        plt.show()

    # filtering of less frequent words
    # here we keep only the top 90% words by occurence
    total = np.sum(counts)
    limit = total * filter_unknown
    cumsum = np.cumsum(counts)
    keep = set(np.array(words)[cumsum <= limit].tolist())

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    for input_text, target_text in sequences:
        new_target_text = []
        for tt in target_text:
            if tt in keep:
                new_target_text.append(tt)
            else:
                new_target_text.append('UNK') # Replace with unknown
        # TODO: reverse the input?
        input_text = list((input_text))
        target_text = new_target_text
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = ['\t'] + target_text + ['\n']
        input_texts.append(input_text)
        target_texts.append(target_text)
        input_characters.update(input_text)
        target_characters.update(target_text)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print(input_characters)
    print('Number of unique output tokens:', num_decoder_tokens)
    print(target_characters)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    # reverse_input_char_index = dict(
    #     (i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data, \
        num_encoder_tokens, num_decoder_tokens, max_decoder_seq_length, target_token_index, input_texts


def get_model(encoder_input_data, decoder_input_data, decoder_target_data,
              num_encoder_tokens, num_decoder_tokens,
              latent_dim, num_samples, model_choice):

    encoder_states = None
    if model_choice == 0:

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        # encoder_inputs = Masking()(encoder_inputs) # Assuming PAD is zeros
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    elif model_choice == 1: # 2 layers LSTM for the encoder

        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder1 = LSTM(latent_dim, recurrent_dropout=0.5, return_sequences=True)
        encoder2 = LSTM(latent_dim, recurrent_dropout=0.5, return_sequences=True)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder2(encoder1(encoder_inputs)))
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    elif model_choice == 2:

        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = GRU(latent_dim, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_gru = GRU(latent_dim, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define sampling models
    if encoder_states is not None:
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
    else:
        encoder_model = None
        decoder_model = None

    return model, encoder_model, decoder_model


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


def train_model(model, batch_size, epochs, encoder_input_data, decoder_input_data,
                decoder_target_data):

    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h = History()
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.000001,
                            verbose=1, epsilon=1e-5)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    callbacks = [rlr, early_stop]
    if is_notebook():
        callbacks.append(PlotLossesKeras())
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks)
    return model, history


def save_model(model, history, model_out, history_out):
    model.save(model_out)
    with open(history_out, 'wb') as f:
        pickle.dump(history.history, f)


def decode_sequence(input_seq, encoder_model, decoder_model,
                    num_decoder_tokens, max_decoder_seq_length, target_token_index):

    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # print(output_tokens)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        # if (sampled_char == '\n' or
        if len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence