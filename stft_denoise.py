import librosa
import pandas as pd
import os
import datetime
import tensorflow as tf
import librosa.display
import glob
import numpy as np
from stft_model import build_model


windowLength = 256
overlap = round(0.25 * windowLength)  # overlap of 75%
ffTLength = windowLength
inputFs = 48e3
fs = 16e3
numFeatures = ffTLength//2 + 1
numSegments = 8


def read_audio(filepath, sample_rate, normalize=True):
    # print(f"Reading: {filepath}").
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
    return audio, sr


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio


def tf_record_parser(record):
    keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
    clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
    noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

    # reshape input and annotation images
    noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (129, 8, 1), name="noise_stft_mag_features")
    clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (129, 1, 1), name="clean_stft_magnitude")
    noise_stft_phase = tf.reshape(noise_stft_phase, (129,), name="noise_stft_phase")

    return noise_stft_mag_features, clean_stft_magnitude


def create_train_val_dataset(path):
    train_tfrecords_filenames = glob.glob(os.path.join(path, 'train_*'))
    np.random.shuffle(train_tfrecords_filenames)
    train_tfrecords_filenames = list(train_tfrecords_filenames)
    val_tfrecords_filenames = glob.glob(os.path.join(path, 'val_*'))

    train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
    train_dataset = train_dataset.map(tf_record_parser)
    train_dataset = train_dataset.shuffle(8192)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(512 + 256)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
    test_dataset = test_dataset.map(tf_record_parser)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(512)
    return train_dataset, test_dataset


if __name__ == '__main__':
    tf.random.set_seed(999)
    np.random.seed(999)
    train_dataset, test_dataset = create_train_val_dataset(path="records")


    # learning_rates = [0.0001, 0.01]
    # epsilons = [0, 0.1, 0.01, 0.001]
    # beta_1 = [0.1, 0.3, 0.5]
    # beta_2 = [0.1, 0.3]
    # smallest_loss = 100
    # parameters = []
    # for l in learning_rates:
    #     for ep in epsilons:
    #         for b1 in beta_1:
    #             for b2 in beta_2:
    #                 model = build_model(l2_strength=0.0, numFeatures=numFeatures, numSegments=numSegments,learning_rate=l, beta_1=b1, beta_2=b2, epsilon=ep)
    #                 early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
    #                                                                            restore_best_weights=True)
    #
    #                 logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #                 tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
    #                 model.fit(train_dataset, steps_per_epoch=600, validation_data=test_dataset, epochs=50)
    #                                 baseline_val_loss = model.evaluate(test_dataset)[0]
    #                 if smallest_loss > baseline_val_loss:
    #                     print(f"learning rate: {l}, beta 1: {b1}, beta 2: {b2}, epsilon: {ep}, loss: {baseline_val_loss}")
    #                     smallest_loss = baseline_val_loss
    #                 parameters.append([l, b1, b2, ep, baseline_val_loss])
    # df = pd.DataFrame(parameters, columns=['learning rate', 'beta_1', 'beta_2', 'epsilon', 'loss'])
    # df.to_csv('parameters.csv', index=False)

    model = build_model(l2_strength=0.0, numFeatures=numFeatures, numSegments=numSegments, learning_rate=0.01, beta_1=0.5,
                        beta_2=0.3, epsilon=0)
    model.load_weights('models/denoiser_cnn_log_mel_generator.h5')
    baseline_val_loss = model.evaluate(test_dataset)[0]
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                               restore_best_weights=True, baseline=baseline_val_loss)
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models/denoiser_cnn_log_mel_generator.h5',
                                                             monitor='val_loss', save_best_only=True)

    model.fit(train_dataset,
              steps_per_epoch=600,
              validation_data=test_dataset,
              epochs=190,
              callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
              )




