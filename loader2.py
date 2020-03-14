from loader import loader, get_noise_filepaths

import automatic_speech_recognition as asr
import numpy as np
import tensorflow as tf
from random import choice
tfkl = tf.keras.layers


_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
# INPUT_LENGTH = 159744  # for data_small
# INPUT_LENGTH = 124621
INPUT_LENGTH = 32000
# INPUT_LENGTH = 1000


def pad_or_trim(x, padded_length=INPUT_LENGTH):
    if len(x) >= padded_length:
        return x[:padded_length]
    return np.hstack((x, np.zeros(padded_length - len(x))))


def load_data(load_noisy=True):
    clean_wavs, noisy_wavs, clean_fps, noisy_fps, transcripts = zip(*loader())
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]

    # pad_or_trim and normalize  ### MUST SHUFFLE AND SPLIT!
    clean_wavs_padded = \
        np.array([pad_or_trim(x) for x in clean_wavs]).astype('float32')
    clean_wavs_padded = clean_wavs_padded / clean_wavs_padded.max()

    noisy_wavs_padded = \
        np.array([pad_or_trim(x) for x in noisy_wavs]).astype('float32')
    noisy_wavs_padded = noisy_wavs_padded / noisy_wavs_padded.max()

    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    enc = asr.text.Alphabet(lang='en')._str_to_label

    # enc = pretrained_pipeline._alphabet._str_to_label
    encoded_transcripts = \
        [[enc[char] for char in label] for label in transcripts]
    encoded_transcripts_padded = \
        np.array([pad_or_trim(x, 91) for x in encoded_transcripts], dtype='float32')

    # return clean_wavs_padded, encoded_transcripts_padded
    return (noisy_wavs_padded, clean_wavs_padded,
            noisy_fps, clean_fps,
            transcripts)


def load_as_tf_dataset(return_transcripts=False):
    # load dataset
    _, clean_audio, _, clean_fps, transcripts = load_data()
    n_samples = len(clean_audio)

    # create tensorflow dataset
    # ds_clean = tf.data.Dataset.from_tensor_slices(clean_audio)
    ds_clean = tf.data.Dataset.from_tensor_slices(clean_audio[:, :INPUT_LENGTH])

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
            # pad_or_trim to length of clean_signal, then randomly offset
            noise = tf.pad(noise, [[0, pad_length]])
            offset = tf.random.uniform((), 0, len(noise) - 1, tf.int32)
            noise = tf.roll(noise, offset, -1, name='post-pad_noise_offset')

        alpha = 0.95
        return alpha * clean_signal + (1 - alpha) * noise

    ds_noisy = ds_clean.map(add_noise)

    if return_transcripts:
        alphabet = asr.text.Alphabet(lang='en')
        encoded_transcripts = alphabet.get_batch_labels(transcripts)
        ds_encoded_transcripts = \
            tf.data.Dataset.from_tensor_slices(encoded_transcripts)
        return ds_clean, ds_noisy, ds_encoded_transcripts, n_samples
    return ds_clean, ds_noisy
