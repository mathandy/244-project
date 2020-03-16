"""This script is contains modified code from the MIT licenced
`automatic_speech_recognition` python module."""

import automatic_speech_recognition as asr
import tensorflow as tf
import os


def weights(lang: str, name: str, version: float):
    """ Model weights are stored in the Google Cloud Bucket. """
    def closure(loader):
        """ The wrapper is required to run downloading after a call. """
        def wrapper() -> asr.pipeline.Pipeline:
            bucket = 'automatic-speech-recognition'
            file_name = f'{lang}-{name}-weights-{version}.h5'
            remote_path = file_name
            local_path = f'{os.path.dirname(__file__)}/models/{file_name}'
            asr.utils.maybe_download_from_bucket(bucket, remote_path, local_path)
            return loader(weights_path=local_path)
        return wrapper
    return closure


@weights(lang='en', name='deepspeech2', version=0.1)
def get_ds2(weights_path: str) -> tf.keras.Model:
    deepspeech2 = asr.model.get_deepspeech2(input_dim=160, output_dim=29,
                                            is_mixed_precision=False)
    deepspeech2.load_weights(weights_path)
    deepspeech2.save_weights(weights_path)
    return deepspeech2


def save_ds2_weights(filepath='ds2_weights.h5'):
    # deepspeech2 = asr.model.get_deepspeech2(input_dim=160, output_dim=29,
    #                                         is_mixed_precision=False)
    # deepspeech2.load_weights(weights_path)
    # pipeline = asr.load('deepspeech2', lang='en')
    ds2 = get_ds2()
    ds2.save_weights(filepath=filepath)


if __name__ == '__main__':
    save_ds2_weights()
