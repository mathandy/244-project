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


def simple_denoiser(x, conv_params=_DEFAULT_CONV_PARAMS):
    x = tfkl.Conv2D(64, 3, name='den_conv0', **conv_params)(x)
    x = tfkl.Conv2D(64, 3, name='den_conv1', **conv_params)(x)
    x = tfkl.Conv2D(1, 3, name='den_conv2', **conv_params)(x)
    return tf.squeeze(x, 3)


# def our_model(pretrained_pipeline, sample_shape):
#
#     inputs = tfkl.Input(shape=sample_shape)
#     # features = pretrained_pipeline._features_extractor(inputs)
#
#     denoised_features = tf.squeeze(simple_denoiser(inputs), -1)
#     # denoised_features = get_generator()(features)
#     # denoised_features = unet(input_size=(256, 256, 1))(features)
#
#     batch_logits = pretrained_pipeline._model(denoised_features)
#
#     model = tf.keras.Model(inputs=inputs, outputs=batch_logits)
#
#     return model


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
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == '__main__':
    # test()

    # load dataset
    # dataset, dev_dataset, shape_of_single_wav = ...
    clean_wavs, noisy_wavs, clean_fps, noisy_fps, transcripts = zip(*loader())

    ### MUST SHUFFLE AND SPLIT!
    def pad(x, l=159744):
        if len(x) == l:
            return x
        return np.hstack((x, np.zeros(l - len(x))))


    # clean_wavs_padded = np.array([pad(x) for x in clean_wavs]).astype('float32')

    ### Seems do not need to pad the wav files
    # normalize
    # max_ = 0
    # for i in range(len(clean_wavs)):
    #     temp_max = clean_wavs[i].max()
    #     if max_ < temp_max:
    #         max_ = temp_max
    # clean_wavs = [x / max_ for x in clean_wavs]
    # clean_wavs_padded = clean_wavs_padded / clean_wavs_padded.max()

    pretrained_pipeline = asr.load('deepspeech2', lang='en')

    enc = pretrained_pipeline._alphabet._str_to_label
    encoded_transcripts = \
        [[enc[char] for char in label] for label in transcripts]
    encoded_transcripts_padded = \
        np.array([pad(x, 91) for x in encoded_transcripts])

    features_clean = pretrained_pipeline._features_extractor(clean_wavs)
    features_noisy = pretrained_pipeline._features_extractor(noisy_wavs)

    train_data_clean = (np.expand_dims(features_clean, -1), encoded_transcripts_padded)
    train_data_noisy = (np.expand_dims(features_noisy, -1), encoded_transcripts_padded)

    features_clean = np.expand_dims(features_clean, -1)
    features_noisy = np.expand_dims(features_noisy, -1)

    # working full pipeline inference example
    # we could use https://www.tensorflow.org/tutorials/customization/autodiff
    batch_size = 5

    model = tf.keras.Sequential([
        tfkl.Conv2D(64, 3, name='den_conv0', padding='SAME'),
        tfkl.Conv2D(64, 3, name='den_conv1', padding='SAME'),
        tfkl.Conv2D(1, 3, name='den_conv2', padding='SAME')
    ])
    # den = simple_denoiser(features_noisy[:batch_size, :, :, None])
    train_feature_noisy = features_noisy[:batch_size]
    train_feature_clean = features_clean[:batch_size]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0000001)
    loss_value, grads = grad(model, train_feature_noisy, train_feature_clean)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("Step: {},          Loss: {}".format(optimizer.iterations.numpy(),
                                               loss(model, train_feature_noisy, train_feature_clean, training=True).numpy()))

    predictions = model(train_feature_noisy)
    tf.nn.softmax(predictions)



    # feature_batch = np.expand_dims(features[:batch_size], -1)
    # ## previous denoiser
    # # den = simple_denoiser(feature_batch)
    # y = pretrained_pipeline._model(den)
    #
    # predictions = pretrained_pipeline.decoder(y)
    # decoded_predictions = \
    #     [[pretrained_pipeline._alphabet._label_to_str[char] for char in l]
    #      for l in predictions]
    # print(decoded_predictions)






    # FIX THIS CODE BELOW
    # m = our_model(pretrained_pipeline, train_data[0].shape[1:])
    # fit(m, pretrained_pipeline, train_data, val_data)
