# from unet import get_unet
# from rnn_generator import get_generator
from loader2 import load_as_tf_dataset, INPUT_LENGTH
from spectrogram import TFSpectrogram
from util import renormalize_quantize_and_save

import os
from time import time
from typing import List
import numpy as np
from scipy.io import wavfile
import automatic_speech_recognition as asr
import tensorflow as tf
tfkl = tf.keras.layers


_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
RUN_NAME = str(int(round(time())))
RESULTS_DIR = os.path.expanduser(
    '~/244-project-results/noisy2probs/' + RUN_NAME)


def get_flat_denoiser():
    model = tf.keras.models.Sequential(layers=[
        tfkl.Lambda(lambda inputs: tf.expand_dims(inputs, -1)),
        tfkl.Conv1D(8, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(8, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(1, 3, name='den_conv2', **_DEFAULT_CONV_PARAMS),
        tfkl.Lambda(lambda outputs: tf.squeeze(outputs, -1))
    ])
    return model


def decode(asr_pipeline, batch_logits):
    decoded_labels = asr_pipeline._decoder(batch_logits)
    predictions = asr_pipeline._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
    """ Get ready features, and make a prediction. """
    features: np.ndarray = self._features_extractor(batch_audio)
    batch_logits = self._model.predict(features, **kwargs)
    decoded_labels = self._decoder(batch_logits)
    predictions = self._alphabet.get_batch_transcripts(decoded_labels)
    return predictions


# def fit(our_model, asr_pipeline, dataset, dev_dataset, **kwargs):
#
#     dataset = asr_pipeline.wrap_preprocess(dataset)
#     dev_dataset = asr_pipeline.wrap_preprocess(dev_dataset)
#     if not our_model.optimizer:  # a loss function and an optimizer
#         y = tfkl.Input(name='y', shape=[None], dtype='int32')
#         loss = asr_pipeline.get_loss()
#         our_model.compile(asr_pipeline._optimizer, loss, target_tensors=[y])
#     tmp = our_model.fit(dataset, validation_data=dev_dataset, **kwargs)
#     print(tmp)
#     return our_model


def get_loss_fcn():
    def get_length(tensor):
        lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
        return tf.cast(lengths, tf.int32)

    def ctc_loss(labels, logits):
        label_length = get_length(labels)
        logit_length = get_length(tf.math.reduce_max(logits, 2))
        return tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                              logits_time_major=False, blank_index=-1)
    return ctc_loss


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # create model
    denoiser_net = get_flat_denoiser()
    deep_speech_v2 = asr.model.deepspeech2.get_deepspeech2(
        input_dim=160, output_dim=29, is_mixed_precision=False)
    deep_speech_v2.load_weights(os.path.join('data', 'ds2_weights.h5'))
    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    # deep_speech_v2 = pretrained_pipeline._model
    for layer in deep_speech_v2.layers:
        layer.trainable = False
    loss_fcn = tf.keras.losses.kld
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    loss_metric = tf.keras.metrics.Mean()

    # load dataset
    ds_clean, ds_noisy, ds_transcripts = load_as_tf_dataset(True)
    # ds_clean = pretrained_pipeline.features_extractor(ds_clean)
    # shuffle and batch data
    ds = tf.data.Dataset.zip((ds_noisy, ds_clean))
    ds = ds.shuffle(buffer_size=4*BATCH_SIZE)
    # ds = ds.shuffle(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    tf_features_extractor = TFSpectrogram(
        audio_length=INPUT_LENGTH,
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning
    )

    # pass ds_clean through asr model
    USE_NUMPY_FEATURES_FOR_CLEAN = False

    if USE_NUMPY_FEATURES_FOR_CLEAN:
        # @tf.numpy_function
        # def np_features_extractor(audio_batch):
        #     return np.array([pretrained_pipeline.features_extractor(x)
        #                      for x in audio_batch])d
        # def get_clean_features(noisy, clean):
        #     return noisy, np_features_extractor(clean)
        raise NotImplementedError
    else:
        def get_clean_features(noisy, clean):
            return noisy, tf_features_extractor(clean)

    def get_asr_probs_for_clean(noisy, clean_features):
        return noisy, deep_speech_v2(clean_features)

    ds = ds.map(get_clean_features)
    ds = ds.map(get_asr_probs_for_clean)

    # train
    for epoch in range(EPOCHS):
        print(f'Start of epoch {epoch}')
        for step, (noisy_audio_batch, clean_probs_batch) in enumerate(ds):

            with tf.GradientTape() as tape:
                denoised_audio_batch = denoiser_net(noisy_audio_batch)
                features = tf_features_extractor(denoised_audio_batch)
                batch_logits = deep_speech_v2(features)
                loss = loss_fcn(clean_probs_batch, batch_logits)

            grads = tape.gradient(loss, denoiser_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, denoiser_net.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))

                # write samples to disk
                prefix = f'epoch-{epoch}_step-{step}_'
                for k, denoised_sample in enumerate(noisy_audio_batch.numpy()):
                    fp = os.path.join(RESULTS_DIR,
                                      prefix + f'sample-{k}_noisy.wav')
                    renormalize_quantize_and_save(denoised_sample, fp)
                for k, denoised_sample in enumerate(denoised_audio_batch.numpy()):
                    fp = os.path.join(RESULTS_DIR,
                                      prefix + f'sample-{k}_denoised.wav')
                    renormalize_quantize_and_save(denoised_sample, fp)


if __name__ == '__main__':
    main()
