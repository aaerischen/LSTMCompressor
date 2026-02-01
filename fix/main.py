import tensorflow as tf
from tensorflow import keras 
import torch
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

os.environ['TF_DETERMINISTIC_OPS'] = '1'

batch_size = 512 # 256
seq_length = 12 # 16
rnn_units = 128 # 256
num_layers = 1 # 2
embedding_size = 192 # 256
start_learning_rate = 0.005 #initial
end_learning_rate = 0.001 #initial
mode = 'both'

path_to_file = "fix\data\enwik5"
path_for_saving = "fix\data\enwik5"
path_to_compressed = path_for_saving + "_compressed.dat"
path_to_decompressed = path_for_saving + "_decompressed.dat"


def build_model(vocab_size: int) -> tf.keras.Model:

    inputs = tf.keras.Input(batch_shape = (batch_size, seq_length), dtype = tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, embedding_size)(inputs)
    for i in range(num_layers):
        return_seq = True if i != num_layers - 1 else False
        x = tf.keras.layers.GRU(rnn_units,
                                 dropout = 0.1,
                                 return_sequences = return_seq,
                                 stateful = True,
                                 recurrent_initializer = 'glorot_uniform'
                                 )(x)
    dense = tf.keras.layers.Dense(vocab_size, 
                                  name='dense_logits'
                                  )(x)
    output = tf.keras.layers.Activation('softmax', 
                                        dtype='float32', 
                                        name='predictions'
                                        )(dense)
    model = tf.keras.Model(inputs=inputs, 
                           outputs=output
                           )

    return model

def reset_seed():
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_symbol(index, length, freq, coder, compress, data):
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data, split):
    with tf.GradientTape() as tape:
        predictions = model(seq_input)
        p = predictions.numpy()
        freqs_batch = np.cumsum(p * 1000000 + 1, axis=1)
        
        symbols = []
        mask = []
        cross_entropy = 0
        denom = 0
        for i in range(batch_size):
            index = pos + i * split
            freq = freqs_batch[i]
            symbol = get_symbol(index, length, freq, coder, compress, data)
            symbols.append(symbol)
            if index < length:
                prob = p[i][symbol]
                if prob <= 0:
                    prob = 1e-6
                cross_entropy += math.log2(prob)
                denom += 1
                mask.append(1.0)
            else:
                mask.append(0.0)

        symbols_tensor = tf.convert_to_tensor(symbols, dtype=tf.int32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
        loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(symbols_tensor, predictions, from_logits = False)
        loss = tf.reduce_sum(loss_per_example * mask_tensor)

        seq_input = tf.concat([seq_input[:, 1:], tf.expand_dims(symbols_tensor, 1)], axis=1)

    gradients = tape.gradient(loss, model.trainable_variables)
    capped_grads = [tf.clip_by_norm(g, 4.0) for g in gradients]
    optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))

    return seq_input, cross_entropy, denom


def process(compress, length, vocab_size, coder, data):
    start = time.time()
    reset_seed()
    model = build_model(vocab_size=vocab_size)
    model.summary()


    split = math.ceil(length / batch_size)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        split,
        end_learning_rate,
        power=1.0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn)

    freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)

    symbols = [get_symbol(i*split, length, freq, coder, compress, data) for i in range(batch_size)]
    seq_input = tf.tile(tf.expand_dims(symbols, 1), [1, seq_length])
    pos = 0
    cross_entropy = 0
    denom = 0

    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'

    while pos < split:
        seq_input, ce, d = train(pos,
                                seq_input,
                                length, 
                                vocab_size, 
                                coder, 
                                model,
                                optimizer, 
                                compress, 
                                data,
                                split)
        cross_entropy += ce
        denom += d
        pos += 1
        if pos % 5 == 0:
            percentage = 100 * pos / split
            print(template.format(percentage, -cross_entropy / denom if denom > 0 else 0, time.time() - start))
    if compress:
        coder.finish()
    print(template.format(100, -cross_entropy / length, time.time() - start))


def compession():
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = math.ceil(len(vocab)/8) * 8

    char2idx = {u: i for i, u in enumerate(vocab)}
    for c in text:
        int_list.append(char2idx[c])
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        out.write(file_len.to_bytes(5, byteorder='big', signed=False))
        for i in range(256):
            bitout.write(1 if i in char2idx else 0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, file_len, vocab_size, enc, int_list)


def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        output = [0] * length
        bitin = BitInputStream(inp)
        vocab = [i for i in range(256) if bitin.read()]
        vocab_size = math.ceil(len(vocab)/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file)/os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()
