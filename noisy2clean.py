# from unet import get_unet
# from rnn_generator import get_generator
from loader import loader, get_noise_filepaths

import automatic_speech_recognition as asr
from scipy.io import wavfile
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from random import choice, randint
tfkl = tf.keras.layers



_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
INPUT_LENGTH = 159744


def pad(x, padded_length=INPUT_LENGTH):
    if len(x) == padded_length:
        return x
    return np.hstack((x, np.zeros(padded_length - len(x))))


def load_data():
    clean_wavs, noisy_wavs, clean_fps, noisy_fps, transcripts = zip(*loader())
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]

    # pad and normalize  ### MUST SHUFFLE AND SPLIT!
    clean_wavs_padded = \
        np.array([pad(x) for x in clean_wavs]).astype('float32')
    clean_wavs_padded = clean_wavs_padded / clean_wavs_padded.max()

    noisy_wavs_padded = \
        np.array([pad(x) for x in noisy_wavs]).astype('float32')
    noisy_wavs_padded = noisy_wavs_padded / noisy_wavs_padded.max()

    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    enc = asr.text.Alphabet(lang='en')._str_to_label

    # enc = pretrained_pipeline._alphabet._str_to_label
    encoded_transcripts = \
        [[enc[char] for char in label] for label in transcripts]
    encoded_transcripts_padded = \
        np.array([pad(x, 91) for x in encoded_transcripts], dtype='float32')

    # return clean_wavs_padded, encoded_transcripts_padded
    return noisy_wavs_padded, clean_wavs_padded


def get_flat_denoiser():
    model = tf.keras.models.Sequential(layers=[
        tfkl.Lambda(lambda inputs: tf.expand_dims(inputs, -1)),
        tfkl.Conv1D(8, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(8, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(1, 3, name='den_conv2', **_DEFAULT_CONV_PARAMS),
        tfkl.Lambda(lambda outputs: tf.squeeze(outputs))
        # tfkl.Flatten(),
        # tfkl.Dense(16000)
    ])
    return model


def noise_generator():
    noise_filepaths = get_noise_filepaths()
    noise_fp = choice(noise_filepaths)
    return


if __name__ == '__main__':

    # load dataset
    _, clean_audio = load_data()

    # create tensorflow dataset
    # ds_noise_fp = tf.data.Dataset.from_generator(noise_generator, tf.string)
    ds_clean = tf.data.Dataset.from_tensor_slices(clean_audio)
    # ds = tf.data.Dataset.zip((ds_clean, ds_clean))

    noise_filepaths = get_noise_filepaths()

    @tf.function
    def add_noise(clean_signal):
        noise, sr = tf.audio.decode_wav(choice(noise_filepaths))
        tf.cast(clean_signal, tf.float32)

        # ensure noise is shorter than clean audio (trim if necessary)
        pad_length = tf.shape(clean_signal)[0] - tf.shape(noise)[0]
        if tf.shape(noise)[0] > tf.shape(clean_signal)[0]:
            noise = noise[:tf.shape(clean_signal)[0]]

        # pad noise to same length as clean audio
        noise = tf.pad(noise, [[0, len(clean_signal) - len(noise)]])

        # shift noise signal
        noise = tf.roll(noise, randint(0, len(noise)-1), -1, name='noise_offset')

        # tf.roll(input, shift, axis, name='noise_offset')
        return 0.5 * (clean_signal + noise)

    ds_noisy = ds_clean.map(add_noise)
    ds = tf.data.Dataset.zip((ds_noisy, ds_clean))
    ds = ds.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # create model
    model = get_flat_denoiser()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    loss_metric = tf.keras.metrics.Mean()
    loss_fcn = tf.keras.losses.MSE

    # train
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,))
        for step, (x_batch, y_batch) in enumerate(ds):
            with tf.GradientTape() as tape:
                yhat_batch = model(x_batch)
                loss = loss_fcn(x_batch, yhat_batch)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
