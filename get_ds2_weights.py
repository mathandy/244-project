import automatic_speech_recognition as asr


def save_ds2_weights(filepath='ds2_weights.h5'):
    pipeline = asr.load('deepspeech2', lang='en')
    pipeline.model.save_weights(filepath=filepath)


if __name__ == '__main__':
    save_ds2_weights()
