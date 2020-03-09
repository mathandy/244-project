# from unet import get_unet
# from rnn_generator import get_generator
from loader import loader

import automatic_speech_recognition as asr
from scipy.io import wavfile
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
import os

_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.00001


def pad(x, l=124621):
    if len(x) == l:
        return x
    return np.hstack((x, np.zeros(l - len(x))))


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
        tfkl.Conv1D(16, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(16, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(16, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        # tfkl.Conv1D(16, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        # tfkl.Conv1D(16, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        # tfkl.Conv1D(16, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        # tfkl.Conv1D(16, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(1, 3, name='den_conv2', **_DEFAULT_CONV_PARAMS),
        tfkl.Lambda(lambda outputs: tf.squeeze(outputs))
        # tfkl.Flatten(),
        # tfkl.Dense(16000)
    ])
    return model



if __name__ == '__main__':

    # load dataset
    x, y = load_data()

    # create tensorflow dataset
    x_ds = tf.data.Dataset.from_tensor_slices(x)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((x_ds, y_ds))
    ds = ds.shuffle(buffer_size=4*BATCH_SIZE).batch(BATCH_SIZE)
    # create model
    model = get_flat_denoiser()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_metric = tf.keras.metrics.Mean()
    # loss_fcn = tf.keras.losses.mean_absolute_percentage_error
    loss_fcn = tf.keras.losses.mean_absolute_error

    # train
    for epoch in range(1, EPOCHS+1):
        print('Start of epoch %d' % (epoch,))
        for step, (x_batch, y_batch) in enumerate(ds):
            with tf.GradientTape() as tape:
                yhat_batch = model(x_batch)
                loss = loss_fcn(y_batch, yhat_batch)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)
            if step % 30 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
            if step == 30:
                path = f"/tmp/244/"
                if not os.path.isdir(path):
                    os.makedirs(path)
                for i in range(5):
                    true_wav = os.path.join(path, f"epoch{epoch}_{i}_true.wav")
                    pred_wav = os.path.join(path, f"epoch{epoch}_{i}_pred.wav")
                    wavfile.write(true_wav, 16000, y_batch[i].numpy())
                    wavfile.write(pred_wav, 16000, yhat_batch[i].numpy())
                    if epoch == 1:
                        input_wav = os.path.join(path, f"{i}_input.wav")
                        wavfile.write(input_wav, 16000, x_batch[i].numpy())

