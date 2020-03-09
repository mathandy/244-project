# from unet import get_unet
# from rnn_generator import get_generator
from loader2 import load_as_tf_dataset

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
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


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


def make_features(self, audio: np.ndarray) -> np.ndarray:
    """ Use `python_speech_features` lib to extract log-spectrogram's. """
    audio = self.normalize(audio.astype(np.float32))
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    audio = self.pad(audio) if self.pad_to else audio
    frames = python_speech_features.sigproc.framesig(
        audio, self.frame_len, self.frame_step, self.winfunc
    )
    features = python_speech_features.sigproc.logpowspec(
        frames, self.frame_len, norm=1
    )
    features = features[:, :self.features_num]  # Cut high frequency part
    return self.standardize(features) if self.is_standardization else features


if __name__ == '__main__':

    # load dataset
    ds_clean, ds_noisy = load_as_tf_dataset()
    ds = tf.data.Dataset.zip((ds_noisy, ds_clean))
    ds = ds.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # create denoiser model
    model = get_flat_denoiser()
    pretrained_pipeline = asr.load('deepspeech2', lang='en')
    loss_fcn = get_loss_fcn()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    loss_metric = tf.keras.metrics.Mean()

    # train
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,))
        for step, (x_batch, y_batch) in enumerate(ds):
            with tf.GradientTape() as tape:
                yhat_batch = model(x_batch)
                features = pretrained_pipeline._features_extractor(x_batch)
                batch_logits = pretrained_pipeline._model(features)
                loss = loss_fcn(x_batch, yhat_batch)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
