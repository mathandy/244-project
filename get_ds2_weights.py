import automatic_speech_recognition as asr

pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.save_weights('ds2_weights.h5')
