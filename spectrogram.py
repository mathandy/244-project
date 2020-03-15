"""A TensorFlow (i.e. gradient-friendly) version of ASR's Spectrogram

This code is modified from parts of MIT licensed modules 
"python_speech_features" and "automatic_speech_recognition"
"""

import decimal
import math
import numpy as np
# import python_speech_features
from automatic_speech_recognition import features
import tensorflow as tf


def log_power_spectrum(frames, fft_length, norm=True):

    complex_spec = tf.signal.rfft(frames, [fft_length])
    magspec = tf.abs(complex_spec)
    ps = 1. / fft_length * tf.square(magspec)

    # ps[ps <= 1e-30] = 1e-30
    ps = tf.clip_by_value(ps, 1e-30, tf.reduce_max(ps))
    lps = 10. * tf.math.log(ps) / tf.math.log(10.)
    if norm:
        return lps - tf.reduce_max(lps)
    else:
        return lps


class TFSpectrogram(features.FeaturesExtractor):

    def __init__(self,
                 audio_length: int,
                 features_num: int,
                 samplerate: int,
                 winlen: float,
                 winstep: float,
                 winfunc=np.hanning,
                 is_standardization=True):
        self.audio_length = audio_length
        self.features_num = features_num
        self.winfunc = winfunc
        self.frame_len = int(winlen * samplerate)
        self.frame_step = int(winstep * samplerate)
        self.is_standardization = is_standardization
        self.window = self.make_window()

    @staticmethod
    def normalize(audio):
        """ Normalize float32 signal to [-1, 1] range. """
        gain = 1. / (tf.reduce_max(tf.abs(audio)) + 1e-5)
        return audio * gain

    @staticmethod
    def standardize(features):
        """ Standardize globally, independently of features. """
        mean, var = tf.nn.moments(features, axes=(1, 2))
        return (features - mean[:, None, None]) / tf.sqrt(var)[:, None, None]

    def make_window(self):
        # Note: `np.hanning(frame_len)` is equivalent to
        # `tf.signal.hann_window(frame_len, periodic=False)`

        def round_half_up(number):
            dec = decimal.Decimal(number)
            return int(dec.quantize(decimal.Decimal('1'),
                                    rounding=decimal.ROUND_HALF_UP))

        frame_len = int(round_half_up(self.frame_len))
        frame_step = int(round_half_up(self.frame_step))
        if self.audio_length <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil(
                (1. * self.audio_length - frame_len) / frame_step))

        win = np.tile(self.winfunc(frame_len), (numframes, 1))

        return tf.constant(win[None, :, :], dtype=tf.float32)

    def make_features(self, audio_batch):
        """ Use `python_speech_features` lib to extract log-spectrogram's. """

        # normalize float32 signal to [-1, 1] range
        audio_batch = self.normalize(audio_batch)

        # re-cast (after normalizing) to 16-bit integers
        # audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        audio_batch = 32767 * audio_batch

        frames = tf.signal.frame(audio_batch, self.frame_len, self.frame_step,
                                 pad_end=True)

        # this line is to match the `python_speech_features` version
        frames = frames[:, :-1, :]

        frames = frames * self.window
        features = log_power_spectrum(frames, self.frame_len, norm=True)

        features = features[:, :, :self.features_num]  # Cut high frequency part

        if self.is_standardization:
            features = self.standardize(features)

        return features

    def __call__(self, audio_batch):
        return self.make_features(audio_batch)


if __name__ == '__main__':
    from loader2 import INPUT_LENGTH
    import automatic_speech_recognition as asr

    pipeline = asr.load('deepspeech2', lang='en')

    audio = np.random.rand(2, INPUT_LENGTH).astype('float32')
    audio = (audio - 0.5) * 2
    audio_copy = audio.copy()

    np_features_extractor = \
        asr.load('deepspeech2', lang='en').features_extractor

    tf_features_extractor = TFSpectrogram(
        audio_length=INPUT_LENGTH,
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning
    )

    np_features = pipeline.features_extractor(audio_copy)
    tf_features = tf_features_extractor(audio)

    print(tf_features.numpy() - np_features)
    from IPython import embed; embed()  ### DEBUG