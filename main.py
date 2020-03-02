# from unet import get_unet
# from rnn_generator import get_generator

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
    for x in sentences:
        print('\n' + x)


def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
    """ Get ready features, and make a prediction. """
    features: np.ndarray = self._features_extractor(batch_audio)
    batch_logits = self._model.predict(features, **kwargs)
    decoded_labels = self._decoder(batch_logits)
    predictions = self._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


def simple_denoiser(x):
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


def our_model(pretrained_pipeline, sample_shape):

    inputs = tfkl.Input(shape=sample_shape)
    features = pretrained_pipeline._features_extractor(inputs)

    denoised_features = simple_denoiser(features)
    # denoised_features = get_generator()(features)
    # denoised_features = unet(input_size=(256, 256, 1))(features)

    batch_logits = pretrained_pipeline._model(denoised_features)

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

    # load dataset
    # dataset, dev_dataset, shape_of_single_wav = ...

    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    # m = our_model(pretrained_pipeline, shape_of_single_wav)
