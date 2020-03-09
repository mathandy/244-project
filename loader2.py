from loader import loader, get_noise_filepaths

import automatic_speech_recognition as asr
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


def load_as_tf_dataset():
    # load dataset
    _, clean_audio = load_data()

    # create tensorflow dataset
    # ds_noise_fp = tf.data.Dataset.from_generator(noise_generator, tf.string)
    ds_clean = tf.data.Dataset.from_tensor_slices(clean_audio)
    # ds = tf.data.Dataset.zip((ds_clean, ds_clean))

    noise_filepaths = get_noise_filepaths()

    # @tf.function
    def add_noise(clean_signal):
        """expects (and returns) 1D tensor"""
        audio_bytes = tf.io.read_file(choice(noise_filepaths))
        noise, sr = tf.audio.decode_wav(audio_bytes)
        noise = tf.squeeze(noise)
        clean_signal = tf.cast(clean_signal, tf.float32)

        # ensure noise is shorter than clean audio (trim if necessary)
        pad_length = tf.shape(clean_signal)[-1] - tf.shape(noise)[-1]
        if pad_length < 0:
            # randomly offset, then trim to length of clean_signal
            offset = tf.random.uniform((), 0, len(noise) - 1, tf.int32)
            noise = tf.roll(noise, offset, -1, name='pre-trim_noise_offset')
            noise = noise[:tf.shape(clean_signal)[0]]
        else:
            # pad to length of clean_signal, then randomly offset
            noise = tf.pad(noise, [[0, pad_length]])
            offset = tf.random.uniform((), 0, len(noise) - 1, tf.int32)
            noise = tf.roll(noise, offset, -1, name='post-pad_noise_offset')

        return 0.5 * (clean_signal + noise)

    ds_noisy = ds_clean.map(add_noise)
    return ds_clean, ds_noisy
