# from unet import get_unet
# from rnn_generator import get_generator
from loader import loader

import automatic_speech_recognition as asr
from scipy.io import wavfile
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers


_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 3


def feature_denoiser(x, conv_params=_DEFAULT_CONV_PARAMS):
    x = tfkl.Conv2D(64, 3, name='den_conv0', **conv_params)(x)
    x = tfkl.Conv2D(64, 3, name='den_conv1', **conv_params)(x)
    x = tfkl.Conv2D(1, 3, name='den_conv2', **conv_params)(x)
    return tf.squeeze(x, 3)


class FlatDenoiser(tf.keras.Model):
    def __init__(self, *args, conv_params=_DEFAULT_CONV_PARAMS, **kwargs):
        super(FlatDenoiser, self).__init__(*args, **kwargs)
        self.conv_params = conv_params
        self.built_layers = None

    def build(self, input_shape):
        self.built_layers = [
            tfkl.Conv1D(8, 3, name='den_conv0', **self.conv_params),
            tfkl.Conv1D(8, 3, name='den_conv1', **self.conv_params),
            tfkl.Conv1D(8, 3, name='den_conv2', **self.conv_params),
            tfkl.Flatten(),
            tfkl.Dense(input_shape[-1])
        ]

    def call(self, inputs, training=True):
        a = tf.expand_dims(inputs, -1)
        for layer in self.built_layers:
            a = layer(a)
        return a


def decode(asr_pipeline, batch_logits):
    decoded_labels = asr_pipeline._decoder(batch_logits)
    predictions = asr_pipeline._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


def test():
    fn = 'test16.wav'  # sample rate 16 kHz, and 16 bit depth
    fs, audio = wavfile.read(fn)  # same as `asr.utils.read_audio()`
    pipeline = asr.load('deepspeech2', lang='en')
    sentences = pipeline.predict([audio])
    for x in sentences:
        print('\n' + x)


def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
    """ Get ready features, and make a prediction. """
    features: np.ndarray = self._features_extractor(batch_audio)
    batch_logits = self._model.predict(features, **kwargs)
    decoded_labels = self._decoder(batch_logits)
    predictions = self._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    loss_object = tf.keras.losses.MSE
    return tf.reduce_sum(loss_object(y_true=y, y_pred=y_))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def align(arrays: list, default=0) -> np.ndarray:
    """ Pad arrays (default along time dimensions). Return the single
    array (batch_size, time, features). """
    max_array = max(arrays, key=len)
    X = np.full(shape=[len(arrays), *max_array.shape],
                fill_value=default, dtype=float)
    for index, array in enumerate(arrays):
        time_dim, features_dim = array.shape
        X[index, :time_dim] = array
    return X


def pad(x, l=159744):
    if len(x) == l:
        return x
    return np.hstack((x, np.zeros(l - len(x))))


def load_data():
    clean_wavs, noisy_wavs, clean_fps, noisy_fps, transcripts = zip(*loader())
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]
    clean_wavs = [wav.astype('float32') for wav in clean_wavs]

    # pad and normalize  ### MUST SHUFFLE AND SPLIT!
    clean_wavs_padded = np.array([pad(x) for x in clean_wavs]).astype('float32')
    clean_wavs_padded = clean_wavs_padded / clean_wavs_padded.max()

    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    enc = asr.text.Alphabet(lang='en')._str_to_label

    # enc = pretrained_pipeline._alphabet._str_to_label
    encoded_transcripts = [[enc[char] for char in label] for label in transcripts]
    encoded_transcripts_padded = \
        np.array([pad(x, 91) for x in encoded_transcripts], dtype='float32')

    return clean_wavs_padded, encoded_transcripts_padded


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



def fit(our_model, asr_pipeline, dataset, dev_dataset, **kwargs):

    dataset = asr_pipeline.wrap_preprocess(dataset)
    dev_dataset = asr_pipeline.wrap_preprocess(dev_dataset)
    if not our_model.optimizer:  # a loss function and an optimizer
        y = tfkl.Input(name='y', shape=[None], dtype='int32')
        loss = asr_pipeline.get_loss()
        our_model.compile(asr_pipeline._optimizer, loss, target_tensors=[y])
    tmp = our_model.fit(dataset, validation_data=dev_dataset, **kwargs)
    print(tmp)
    return our_model


if __name__ == '__main__':

    # load dataset
    x, y = load_data()

    # create tensorflow dataset
    x_ds = tf.data.Dataset.from_tensor_slices(x)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((x_ds, y_ds))
    ds = ds.shuffle(buffer_size=1024).batch(32)

    # create denoiser model
    model = get_flat_denoiser()
    pretrained_pipeline = asr.load('deepspeech2', lang='en')


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
