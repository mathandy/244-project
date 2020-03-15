import automatic_speech_recognition as asr
from scipy.io import wavfile


def test_asr_pipeline():
    fn = 'test16.wav'  # sample rate 16 kHz, and 16 bit depth
    fs, audio = wavfile.read(fn)  # same as `asr.utils.read_audio()`
    pipeline = asr.load('deepspeech2', lang='en')
    sentences = pipeline.predict([audio])
    for x in sentences:
        print('\n' + x)


if __name__ == '__main__':
    test_asr_pipeline()
