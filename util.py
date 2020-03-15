import numpy as np
from scipy.io import wavfile


def renormalize_quantize_and_save(audio, fp):
    gain = 1. / (np.max(np.abs(audio)) + 1e-5)
    audio = audio * gain
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(fp, 16000, audio)
