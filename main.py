from unet import get_unet

import automatic_speech_recognition as asr
from scipy.io import wavfile
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers


def test():
    fn = 'test16.wav'  # sample rate 16 kHz, and 16 bit depth
    fs, audio = wavfile.read(fn)  # same as `asr.utils.read_audio()`
    pipeline = asr.load('deepspeech2', lang='en')
    sentences = pipeline.predict([audio])
    print(sentences)

# batch_audio = [audio]
# m = pipeline._model
#
# # Parameters for batch normalization.
# _BATCH_NORM_EPSILON = 1e-5
# _BATCH_NORM_DECAY = 0.997
#
# # Filters of convolution layer
# _CONV_FILTERS = 32


def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
    """ Get ready features, and make a prediction. """
    features: np.ndarray = self._features_extractor(batch_audio)
    batch_logits = self._model.predict(features, **kwargs)
    decoded_labels = self._decoder(batch_logits)
    predictions = self._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


# def get_generator(input_static_dim=59,
#                   input_dynamic_dim=118,
#                   num_hidden=5,
#                   hidden_dim=512,
#                   dropout_rate=0.5):
#     """Credit: from MIT Licensed
#         https://github.com/tkm2261/dnn-voice-changer
#     """
#
#     inputs_all = x = tfkl.Input(shape=(input_static_dim + input_dynamic_dim,))
#     x_static = tfkl.Lambda(lambda x: x[:, :input_static_dim])(inputs_all)
#
#     t_x = tfkl.Dense(input_static_dim, activation='sigmoid')(x_static)
#
#     for _ in range(num_hidden):
#         act = tfkl.LeakyReLU()
#         x = tfkl.Dense(hidden_dim, activation=act)(x)
#         x = tfkl.Dropout(dropout_rate)(x)
#
#     g_x = tfkl.Dense(input_static_dim)(x)
#
#     outputs = tfkl.add([x_static, tfkl.multiply([t_x, g_x])])
#
#     model = tf.keras.Model(inputs=inputs_all, outputs=outputs, name='generator')
#
#     return model


def simple_denoiser(x):
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


def our_model(pretrained_pipeline, sample_shape):
    # denoiser = get_generator()
    # denoiser = unet(input_size=(256, 256, 1))

    inputs = tfkl.Input(shape=sample_shape)
    denoised_audio = simple_denoiser(inputs)
    features = pretrained_pipeline._features_extractor(denoised_audio)

    simple_denoiser()
    batch_logits = pretrained_pipeline._model.predict(features)

    model = tf.keras.Model(inputs=inputs, outputs=batch_logits)

    return model


def decode(asr_pipeline, batch_logits):
    decoded_labels = asr_pipeline._decoder(batch_logits)
    predictions = asr_pipeline._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


def fit(our_model, asr_pipeline, dataset, dev_dataset, **kwargs):

    dataset = asr_pipeline.wrap_preprocess(dataset)
    dev_dataset = asr_pipeline.wrap_preprocess(dev_dataset)
    if not our_model.optimizer:  # a loss function and an optimizer
        y = tfkl.Input(name='y', shape=[None], dtype='int32')
        loss = asr_pipeline.get_loss()
        our_model.compile(asr_pipeline._optimizer, loss, target_tensors=[y])
    return our_model.fit(dataset, validation_data=dev_dataset, **kwargs)


if __name__ == '__main__':
    test()
