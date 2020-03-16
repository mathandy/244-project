# from unet import get_unet
# from rnn_generator import get_generator
from timit_loader import load_timit_as_tf_dataset
from spectrogram import TFSpectrogram
from util import renormalize_quantize_and_save

import os
from os.path import join as fpath
from typing import List
import numpy as np
import automatic_speech_recognition as asr
import tensorflow as tf
tfkl = tf.keras.layers


_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}


def get_flat_denoiser(params=_DEFAULT_CONV_PARAMS):
    model = tf.keras.models.Sequential(layers=[
        tfkl.Lambda(lambda inputs: tf.expand_dims(inputs, -1)),
        tfkl.Conv1D(64, 13, name='den_conv0', **params),
        tfkl.Conv1D(64, 13, name='den_conv1', **params),
        tfkl.Conv1D(64, 13, name='den_conv2', **params),
        tfkl.Conv1D(1, 13, name='den_conv3', **params),
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
    ds_clean, ds_noisy, ds_transcripts, n_samples = \
        load_timit_as_tf_dataset(input_lenth=args.input_length,
                                 noisiness=args.noisiness)
    ds = tf.data.Dataset.zip((ds_noisy, ds_transcripts, ds_clean))
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
    os.makedirs(args.log_dir, exist_ok=True)

    ds_train, ds_val, ds_test = prepare_data(args)

    # create model
    denoiser_net = get_flat_denoiser()
    deep_speech_v2 = asr.model.deepspeech2.get_deepspeech2(
        input_dim=160, output_dim=29, is_mixed_precision=False)
    # these pretrained weights come from `automatic-speech-recognition` module
    deep_speech_v2.load_weights('ds2_weights.h5')
    # pretrained_pipeline = asr.load('deepspeech2', lang='en')
    # deep_speech_v2 = pretrained_pipeline._model
    for layer in deep_speech_v2.layers:
        layer.trainable = False
    features_extractor = TFSpectrogram(
        audio_length=args.input_length,
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning
    )

    loss_fcn = get_loss_fcn()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)
    train_loss_ctc = tf.keras.metrics.Mean('train_loss_ctc', dtype=tf.float32)
    train_loss_mae = tf.keras.metrics.Mean('train_loss_mae', dtype=tf.float32)
    val_loss_ctc = tf.keras.metrics.Mean('val_loss_ctc', dtype=tf.float32)
    val_loss_mae = tf.keras.metrics.Mean('val_loss_mae', dtype=tf.float32)
    test_loss_ctc = tf.keras.metrics.Mean('test_loss_ctc', dtype=tf.float32)
    test_loss_mae = tf.keras.metrics.Mean('test_loss_mae', dtype=tf.float32)
    ctc_weight = tf.constant(args.ctc_weight, dtype=tf.float32)
    mae_weight = tf.constant(args.mae_weight, dtype=tf.float32)

    # for tensorboard
    train_summary_writer = tf.summary.create_file_writer(
        fpath(args.log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        fpath(args.log_dir, 'val'))

    # train
    for epoch in range(args.epochs):
        print(f'Start of epoch {epoch}')
        for step, (noisy_audio_batch, enc_transcript_batch, clean_audio_batch) \
                in enumerate(ds_train):

            with tf.GradientTape() as tape:
                denoised_audio_batch = denoiser_net(noisy_audio_batch)
                features = features_extractor(denoised_audio_batch)
                batch_logits = deep_speech_v2(features)
                loss_ctc = loss_fcn(enc_transcript_batch, batch_logits)
                loss_mae = tf.keras.losses.mae(clean_audio_batch,
                                               denoised_audio_batch)
                loss = ctc_weight*loss_ctc + mae_weight*loss_mae

            grads = tape.gradient(loss, denoiser_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, denoiser_net.trainable_weights))

            train_loss_ctc(loss_ctc)
            train_loss_mae(loss_mae)

            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'train_loss_ctc', train_loss_ctc.result(), step=epoch)
                tf.summary.scalar(
                    'train_loss_mae', train_loss_mae.result(), step=epoch)

            if step % 100 == 0:
                print('step %s: (train) ctc loss = %s'
                      '' % (step, train_loss_ctc.result()))
                print('step %s: (train) mae loss = %s'
                      '' % (step, train_loss_mae.result()))

                # write samples to disk
                prefix = f'epoch-{epoch}_step-{step}_'
                for k, sample in enumerate(noisy_audio_batch.numpy()):
                    fp = fpath(args.results_dir,
                               prefix + f'sample-{k}_input.wav')
                    renormalize_quantize_and_save(sample, fp)
                for k, sample in enumerate(denoised_audio_batch.numpy()):
                    fp = fpath(args.results_dir,
                               prefix + f'sample-{k}_denoised.wav')
                    renormalize_quantize_and_save(sample, fp)

        # validate
        for step, (noisy_audio_batch, enc_transcript_batch, clean_audio_batch) \
                in enumerate(ds_val):
            denoised_audio_batch = denoiser_net(noisy_audio_batch)
            features = features_extractor(denoised_audio_batch)
            batch_logits = deep_speech_v2(features)
            loss_ctc = loss_fcn(enc_transcript_batch, batch_logits)
            loss_mae = tf.keras.losses.mae(clean_audio_batch,
                                           denoised_audio_batch)
            # loss = ctc_weight * loss_ctc + mae_weight * loss_mae

            val_loss_ctc(loss_ctc)
            val_loss_mae(loss_mae)
            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss_ctc', val_loss_ctc.result(), step=epoch)
                tf.summary.scalar('val_loss_mae', val_loss_mae.result(), step=epoch)

        print(f'epoch {epoch}: val ctc loss:', val_loss_ctc.result())
        print(f'epoch {epoch}: val mae loss:', val_loss_mae.result())

        # Reset metrics every epoch
        val_loss_ctc.reset_states()
        val_loss_mae.reset_states()
        train_loss_ctc.reset_states()
        train_loss_mae.reset_states()

    # test
    counter = 0
    for step, (noisy_audio_batch, enc_transcript_batch, clean_audio_batch) in \
            enumerate(ds_test):
        denoised_audio_batch = denoiser_net(noisy_audio_batch)
        features = features_extractor(denoised_audio_batch)
        batch_logits = deep_speech_v2(features)
        loss_ctc = loss_fcn(enc_transcript_batch, batch_logits)
        loss_mae = tf.keras.losses.mae(clean_audio_batch,
                                       denoised_audio_batch)
        # loss = ctc_weight * loss_ctc + mae_weight * loss_mae

        test_loss_ctc(loss_ctc)
        test_loss_mae(loss_mae)

        # write samples to disk
        prefix = f'test_'
        for clean_sample, denoised_sample in \
                zip(noisy_audio_batch.numpy(), denoised_audio_batch.numpy()):
            fp = fpath(args.results_dir, prefix + f'{counter}_input.wav')
            renormalize_quantize_and_save(clean_sample, fp)

            fp = fpath(args.results_dir, prefix + f'{counter}_denoised.wav')
            renormalize_quantize_and_save(denoised_sample, fp)
            counter += 1
    print("Test CTC Loss:", test_loss_ctc.result())
    print("Test MAE Loss:", test_loss_mae.result())


if __name__ == '__main__':
    from parameters import get_run_parameters
    from timit_loader import TIMIT_DATA_DIR, NO_TIMIT_DATA_ERROR_MESSAGE

    if not os.path.exists(os.path.join(TIMIT_DATA_DIR, "TRAIN")):
        raise FileNotFoundError(NO_TIMIT_DATA_ERROR_MESSAGE)

    custom_static_params = {
        'run_type': 'noisy2txt',
        'input_length': 32000,
        'noisiness': 0.5,
        'epochs': 100,
        'batch_size': 1,
        'mae_weight': 1.,
        'ctc_weight': 1.,
        'results_dir_root': 'results',
        'log_dir_root': 'logs',
        # 'results_dir_root': os.path.expanduser('~/244-project-results'),
        # 'log_dir_root': os.path.expanduser('~/244-project-logs')
    }

    custom_dynamic_params = {}

    run_args = get_run_parameters(custom_static_params, custom_dynamic_params)
    main(run_args)
