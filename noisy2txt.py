# from unet import get_unet
# from rnn_generator import get_generator
from loader2 import load_as_tf_dataset, INPUT_LENGTH
from spectrogram import TFSpectrogram
from util import renormalize_quantize_and_save

import os
from os.path import join as fpath
from time import time
from typing import List
import numpy as np
import automatic_speech_recognition as asr
import tensorflow as tf
tfkl = tf.keras.layers


def get_flat_denoiser():
    from _params import _DEFAULT_CONV_PARAMS
    model = tf.keras.models.Sequential(layers=[
        tfkl.Lambda(lambda inputs: tf.expand_dims(inputs, -1)),
        tfkl.Conv1D(64, 13, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(64, 13, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(64, 13, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(1, 13, name='den_conv2', **_DEFAULT_CONV_PARAMS),
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


def prepare_data(args):
    # load and shuffle dataset
    ds_clean, ds_noisy, ds_transcripts, n_samples = load_as_tf_dataset(True)
    ds = tf.data.Dataset.zip((ds_noisy, ds_transcripts))
    ds = ds.shuffle(buffer_size=args.shuffle_buffer_size)

    # split dataset
    # https://stackoverflow.com/questions/51125266
    train_size = int(0.7 * n_samples)
    test_size = int(0.15 * n_samples)

    ds_train = ds.take(train_size)
    ds_test = ds.skip(train_size)
    ds_val = ds_test.skip(test_size)
    ds_test = ds_test.take(test_size)

    # batch dataset
    ds_train = ds_train.batch(args.batch_size)
    ds_val = ds_val.batch(args.batch_size)
    ds_test = ds_test.batch(args.batch_size)
    return ds_train, ds_val, ds_test


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    ds_train, ds_val, ds_test = prepare_data(args)

    # create model
    denoiser_net = get_flat_denoiser()
    deep_speech_v2 = asr.model.deepspeech2.get_deepspeech2(
        input_dim=160, output_dim=29, is_mixed_precision=False)
    deep_speech_v2.load_weights(fpath('data', 'ds2_weights.h5'))
    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    # deep_speech_v2 = pretrained_pipeline._model
    for layer in deep_speech_v2.layers:
        layer.trainable = False
    features_extractor = TFSpectrogram(
        audio_length=INPUT_LENGTH,
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning
    )

    loss_fcn = get_loss_fcn()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    # for tensorboard
    train_summary_writer = tf.summary.create_file_writer(
        fpath(args.log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        fpath(args.log_dir, 'val'))

    # train
    for epoch in range(args.epochs):
        print(f'Start of epoch {epoch}')
        for step, (audio_batch, encoded_transcript_batch) in enumerate(ds_train):

            with tf.GradientTape() as tape:
                denoised_audio_batch = denoiser_net(audio_batch)
                features = features_extractor(denoised_audio_batch)
                batch_logits = deep_speech_v2(features)
                loss = loss_fcn(encoded_transcript_batch, batch_logits)

            grads = tape.gradient(loss, denoiser_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, denoiser_net.trainable_weights))

            train_loss(loss)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, train_loss.result()))

                # write samples to disk
                prefix = f'epoch-{epoch}_step-{step}_'
                for k, sample in enumerate(audio_batch.numpy()):
                    fp = fpath(args.results_dir,
                               prefix + f'sample-{k}_input.wav')
                    renormalize_quantize_and_save(sample, fp)
                for k, sample in enumerate(denoised_audio_batch.numpy()):
                    fp = fpath(args.results_dir,
                               prefix + f'sample-{k}_denoised.wav')
                    renormalize_quantize_and_save(sample, fp)

        # validate
        for step, (audio_batch, encoded_transcript_batch) in enumerate(ds_val):
            denoised_audio_batch = denoiser_net(audio_batch)
            features = features_extractor(denoised_audio_batch)
            batch_logits = deep_speech_v2(features)
            loss = loss_fcn(encoded_transcript_batch, batch_logits)

            val_loss(loss)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)

            print('epoch train loss:', train_loss.result())
            print('epoch val loss:', val_loss.result())

        # Reset metrics every epoch
        train_loss.reset_states()
        val_loss.reset_states()

    # test
    counter = 0
    for step, (audio_batch, encoded_transcript_batch) in enumerate(ds_test):
        denoised_audio_batch = denoiser_net(audio_batch)
        features = features_extractor(denoised_audio_batch)
        batch_logits = deep_speech_v2(features)
        test_loss = loss_fcn(encoded_transcript_batch, batch_logits)
        print("Test Loss:", test_loss)

        # write samples to disk
        prefix = f'test_'
        for clean_sample, denoised_sample in \
                zip(audio_batch.numpy(), denoised_audio_batch.numpy()):
            fp = fpath(args.results_dir, prefix + f'{counter}_input.wav')
            renormalize_quantize_and_save(clean_sample, fp)

            fp = fpath(args.results_dir, prefix + f'{counter}_denoised.wav')
            renormalize_quantize_and_save(denoised_sample, fp)
            counter += 1


if __name__ == '__main__':
    from parameters import get_run_parameters
    run_args = get_run_parameters({'run_type': 'noisy2txt'})
    main(run_args)
