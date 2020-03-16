import automatic_speech_recognition as asr
from automatic_speech_recognition.load import weights
import tensorflow as tf


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
