import os.path
from collections import namedtuple
from random import randint

import numpy as np
from SmilesEnumerator import SmilesEnumerator
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Masking, Embedding
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from livelossplot import PlotLossesKeras
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

EncodedResult = namedtuple('EncodedResult', 'vocab, vocab_to_idx, idx_to_vocab, int_encoded, onehot_encoded, max_features, max_timesteps')

def get_sequence_in(n_timesteps_in, n_features_in):
    sequence_in = [randint(0, n_features_in-1) for _ in range(n_timesteps_in)]
    sequence_in = np.array([randint(0, n_features_in-1) for _ in range(n_timesteps_in)])
    return sequence_in


def get_sequence_out(sequence_in, n_timesteps_out, n_features_out, delete, multiply):
    sequence_out = []
    temp = [chr(x) for x in sequence_in + 65]
    for x in temp:
        if len(sequence_out) >= n_timesteps_out:
            break
        if np.random.random() > delete: # keep this character
            expanded = [x] * randint(1, multiply) # multiply this character randomly in length
            for ch in expanded: # and insert up to n_timesteps_out
                if len(sequence_out) <= n_timesteps_out:
                    sequence_out.append(ch)
        else: # got deleted, move on
            continue
    sequence_out = ['\t'] + sequence_out + ['\n']
    sequence_out = np.array(sequence_out)
    return sequence_out


def generate_sequences(n_samples, n_timesteps_in, n_timesteps_out, n_features_in, n_features_out,
                       delete=0.0, multiply=1, permute=0):
    X = []
    y = []
    count = 0
    while count < n_samples:
        sequence_in = get_sequence_in(n_timesteps_in, n_features_in)
        sequence_out = get_sequence_out(sequence_in, n_timesteps_out, n_features_out, delete, multiply)
        sequence_in = np.array(list(map(str, sequence_in.tolist())))
        if len(sequence_out) > 0:
            if permute == 0:
                X.append(sequence_in)
                y.append(sequence_out)
                count += 1
            else:
                for i in range(permute):
                    first = [sequence_out[0]]
                    last = [sequence_out[-1]]
                    between = sequence_out[1:-1]
                    sequence_out = np.concatenate([first, np.random.permutation(between), last])
                    X.append(sequence_in)
                    y.append(sequence_out)
                    count += 1
    X = np.array(X)
    y = np.array(y)
    return X, y


def load_data(data_path, num_samples, repeat=0):
    sme = SmilesEnumerator()
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
        X = []
        y = []
        for line in lines[: min(num_samples, len(lines) - 1)]:
            if line in ['\n', '\r\n']:
                continue

            original_sequence_in, sequence_out = line.split('\t')
            sequence_in = list(original_sequence_in)
            sequence_out = ['\t'] + sequence_out.split(',') + ['\n']
            sequence_in = np.array(list(filter(None, sequence_in)))
            sequence_out = np.array(list(filter(None, sequence_out)))
            X.append(sequence_in)
            y.append(sequence_out)

            if repeat > 0:
                for i in range(repeat-1):
                    try:
                        sequence_in = sme.randomize_smiles(original_sequence_in)
                        sequence_in = list(sequence_in)
                        sequence_in = np.array(list(filter(None, sequence_in)))
                        X.append(sequence_in)
                        y.append(sequence_out)
                    except AttributeError:
                        continue

        X = np.array(X)
        y = np.array(y)
        return X, y


def get_vocab(my_arr):
    try:
        vocab = sorted(list(set(my_arr.flatten().tolist())))
    except TypeError:
        vocab = sorted(list(set([val for row in my_arr.flatten() for val in row])))
    vocab = list(map(str, vocab))
    vocab = ['PAD'] + vocab
    integer_encoded = list(range(len(vocab)))
    vocab_to_idx = dict(zip(vocab, integer_encoded))
    idx_to_vocab = dict(zip(integer_encoded, vocab))
    print('vocab (%d) %s' % (len(vocab), vocab))
    # print('integer_encoded', integer_encoded)
    # print('vocab_to_idx', vocab_to_idx)
    # print('idx_to_vocab', idx_to_vocab)
    return vocab, vocab_to_idx, idx_to_vocab


def int_encode(my_arr, vocab_to_idx, n_timesteps):
    new_arr = []
    for row in my_arr:
        encoded = [vocab_to_idx[x] for x in row]
        new_arr.append(encoded)
    padded = pad_sequences(new_arr, maxlen=n_timesteps)
    new_arr = np.array(padded)
    return new_arr


def encode(data, encoded_result=None):
    if encoded_result is None:
        vocab, vocab_to_idx, idx_to_vocab = get_vocab(data)
        max_features = len(vocab)
        max_timesteps = max([len(n) for n in data])
        int_encoded = int_encode(data, vocab_to_idx, max_timesteps)
        onehot_encoded = to_categorical(int_encoded, num_classes=max_features)
    else:
        vocab = encoded_result.vocab
        vocab_to_idx = encoded_result.vocab_to_idx
        idx_to_vocab = encoded_result.idx_to_vocab
        max_features = encoded_result.max_features
        max_timesteps = encoded_result.max_timesteps
        int_encoded = int_encode(data, vocab_to_idx, max_timesteps)
        onehot_encoded = to_categorical(int_encoded, num_classes=max_features)
    res = EncodedResult(vocab, vocab_to_idx, idx_to_vocab,
                        int_encoded, onehot_encoded,
                        max_features, max_timesteps)
    return res


# define model
def get_model(model_choice, latent_dim,
              n_timesteps_in, n_timesteps_out,
              n_features_in, n_features_out,
              dropout=0.0, recurrent_dropout=0.0,
              weights_file=None):

    additional = {}
    if model_choice == 0:

        # a simple baseline encoder-decoder model
        # see https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
        model = Sequential()
        model.add(Masking(mask_value=0, input_shape=(n_timesteps_in, n_features_in)))
        model.add(LSTM(latent_dim))
        model.add(RepeatVector(n_timesteps_out))
        model.add(LSTM(latent_dim, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features_out, activation='softmax')))

    elif model_choice == 1:

        # the seq2seq model as described in
        # - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

        # TODO: also see for future reference:
        # - https://www.bountysource.com/issues/40722225-how-to-add-attention-on-top-of-a-recurrent-layer-text-classification
        # - https://github.com/philipperemy/keras-attention-mechanism

        # define training encoder, accepts one-hot encoding
        encoder_inputs = Input(shape=(None, n_features_in))
        encoder_mask = Masking(mask_value=0)(encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
        encoder_outputs, state_h, state_c = encoder(encoder_mask)
        encoder_states = [state_h, state_c]

        # define training decoder, accepts one-hot encoding
        decoder_inputs = Input(shape=(None, n_features_out))
        decoder_mask = Masking(mask_value=0)(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_mask,
                                             initial_state=encoder_states)
        decoder_dense = Dense(n_features_out, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)

        # define inference decoder
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

        additional['encoder_model'] = encoder_model
        additional['decoder_model'] = decoder_model

    elif model_choice == 2:

        # similar to the seq2seq model above, but using embedding layers
        # see:
        # - https://stackoverflow.com/questions/49477097/keras-seq2seq-word-embedding
        # - https://stackoverflow.com/questions/50305574/keras-seq2seq-padding
        # - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        #   (What if I want to use a word-level model with integer sequences?)

        # define training encoder, accepts sequences which will be embedded
        embedding_size = int(latent_dim / 2)
        encoder_inputs = Input(shape=(n_timesteps_in,))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_embedding = Embedding(output_dim=embedding_size, input_dim=n_features_in, mask_zero=True)
        encoder_embedding_context = encoder_embedding(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder(encoder_embedding_context)
        encoder_states = [state_h, state_c]

        # define training decoder, accepts sequences
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(output_dim=embedding_size, input_dim=n_features_out, mask_zero=True)
        decoder_embedding_context = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding_context, initial_state=encoder_states)
        decoder_dense = Dense(n_features_out, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # the canonical seq2seq model as described in
        # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)

        # define inference decoder
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding_context, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        additional['encoder_model'] = encoder_model
        additional['decoder_model'] = decoder_model

    elif model_choice == 3:

        # see https://github.com/farizrahman4u/seq2seq/
        model = AttentionSeq2Seq(output_dim=n_features_out,
                      hidden_dim=latent_dim,
                      output_length=n_timesteps_out,
                      input_shape=(n_timesteps_in, n_features_in))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if weights_file is not None:
        print('Loading model weights from', weights_file)
        model.load_weights(weights_file)

    return model, additional


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


def train_model(model, encoded_X, encoded_y, model_choice,
                latent_dim, batch_size, epochs,
                n_timesteps_in, n_timesteps_out, n_features_in, n_features_out):

    def get_decoder_target_data(decoder_input_data):
        decoder_target_data = np.zeros_like(decoder_input_data)
        for i in range(len(decoder_input_data)):
            target_text = decoder_input_data[i]
            for t, char in enumerate(target_text):
                if t > 0:
                    # decoder_target_data will be ahead by one timestep and not include the start character.
                    decoder_target_data[i, t - 1] = char
        return decoder_target_data

    if model_choice == 0:
        encoder_input_data = encoded_X.onehot_encoded
        decoder_input_data = encoded_y.onehot_encoded
        decoder_target_data = get_decoder_target_data(decoder_input_data)
        X = encoder_input_data
        y = decoder_target_data
    elif model_choice == 1:
        encoder_input_data = encoded_X.onehot_encoded
        decoder_input_data = encoded_y.onehot_encoded
        decoder_target_data = get_decoder_target_data(decoder_input_data)
        X = [encoder_input_data, decoder_input_data]
        y = decoder_target_data
    elif model_choice == 2:
        encoder_input_data = encoded_X.int_encoded
        decoder_input_data = encoded_y.int_encoded
        decoder_target_data = get_decoder_target_data(encoded_y.onehot_encoded)
        X = [encoder_input_data, decoder_input_data]
        y = decoder_target_data
    elif model_choice == 3:
        X = encoded_X.onehot_encoded
        y = encoded_y.onehot_encoded

    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.000001,
                            verbose=1, epsilon=1e-5)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    callbacks = [rlr, early_stop]
    if is_notebook():
        callbacks.append(PlotLossesKeras())

    validation_split = 0.2
    model.fit(X, y, batch_size=batch_size,
              epochs=epochs, validation_split=validation_split, callbacks=callbacks)


def evaluate_model(X_new, y_new, encoded_X, encoded_y, model, model_choice, additional, sep=','):
    encoded_X_new = encode(X_new, encoded_result=encoded_X)
    encoded_y_new = encode(y_new, encoded_result=encoded_y)
    if model_choice == 0:
        evaluate_model_0(X_new, y_new, encoded_X_new, encoded_y_new, model, sep)
    elif model_choice == 1:
        evaluate_model_1(X_new, y_new, encoded_X_new, encoded_y_new, additional, sep)
    elif model_choice == 2:
        evaluate_model_2(X_new, y_new, encoded_X_new, encoded_y_new, additional, sep)


def evaluate_model_0(X_new, y_new, encoded_X_new, encoded_y_new, model, sep):

    y_hat_new = model.predict(encoded_X_new.onehot_encoded)
    idx_to_vocab = encoded_y_new.idx_to_vocab

    correct = 0
    for i in range(len(y_hat_new)):
        predicted = y_hat_new[i]
        max_idx = [np.argmax(i) for i in predicted]
        y_hat = np.array([idx_to_vocab[i] for i in max_idx])
        query = sep.join(X_new[i])
        actual = sep.join(y_new[i][1:]).strip()
        predicted = []
        for pred in y_hat:
            if pred == 'PAD':
                continue
            if pred == '\n':
                break
            predicted.append(pred)
        predicted = sep.join(predicted).strip()
        print('Query    ', query)
        print('Actual   ', actual)
        print('Predicted', predicted)
        print(max_idx)
        print()
        if actual == predicted:
            correct += 1
    print('correct', correct)


def evaluate_model_1(X_new, y_new, encoded_X_new, encoded_y_new, additional, sep):

    vocab_to_idx_out = encoded_y_new.vocab_to_idx
    n_features_out = encoded_y_new.max_features
    n_timesteps_out = encoded_y_new.max_timesteps

    encoder_model = additional['encoder_model']
    decoder_model = additional['decoder_model']
    correct = 0
    for i in range(len(y_new)):
        input_seq = encoded_X_new.onehot_encoded[i]
        input_seq = input_seq.reshape(1, input_seq.shape[0], input_seq.shape[1])
        y_hat = decode_sequence(input_seq, encoder_model, decoder_model,
                                           n_features_out, n_timesteps_out, vocab_to_idx_out)
        query = sep.join(X_new[i])
        actual = sep.join(y_new[i][1:-1])
        predicted = sep.join(list(filter(lambda x: x not in ['PAD', '\t', '\n'], y_hat)))
        print('Query     "' + query + '"')
        print('Actual    "' + str(actual) + '"')
        print('Predicted "' + str(predicted) + '"')
        print(y_hat)
        print()
        if actual == predicted:
            correct += 1
    print('correct', correct)


def evaluate_model_2(X_new, y_new, encoded_X_new, encoded_y_new, additional, sep):

    vocab_to_idx_out = encoded_y_new.vocab_to_idx
    n_features_out = encoded_y_new.max_features
    n_timesteps_out = encoded_y_new.max_timesteps

    encoder_model = additional['encoder_model']
    decoder_model = additional['decoder_model']
    correct = 0
    for i in range(len(y_new)):
        input_seq = encoded_X_new.int_encoded[i]
        input_seq = input_seq.reshape(1, input_seq.shape[0])

        # TODO: doesn't work!!
        # y_hat = decode_sequence(input_seq, encoder_model, decoder_model,
        #                                    n_features_out, n_timesteps_out, vocab_to_idx_out)
        # query = sep.join(X_new[i])
        # actual = sep.join(y_new[i][1:-1])
        # predicted = sep.join(list(filter(lambda x: x not in ['PAD', '\t', '\n'], y_hat)))
        # print('Query     "' + query + '"')
        # print('Actual    "' + str(actual) + '"')
        # print('Predicted "' + str(predicted) + '"')
        # print(y_hat)
        # print()
        # if actual == predicted:
        #     correct += 1

    print('correct', correct)

# TODO: implement beam search https://stackoverflow.com/questions/48799623/speeding-up-beam-search-with-seq2seq
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
        if (sampled_char == '\n') or (len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# TODO: https://github.com/farizrahman4u/seq2seq/issues/54
def save_model(model, history, model_out, history_out):
    model.save(model_out)
    with open(history_out, 'wb') as f:
        pickle.dump(history.history, f)