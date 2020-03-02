import os


def get_file_info(filename):
    with open(filename, 'r') as f:
        file_info = []
        for line in f.read().split('\n'):
            if not line.strip():
                continue
            fn1, junk, label = line.strip().split(',')
            file_info.append((fn1.strip(), junk.strip(), label.strip()))
    return file_info


def loader():
    """Return a 6 (unshuffled) tf dataset objects:
        clean_audio, clean_audio_filepath, noisy_audio, noisy_audio_filepath, noise type, label

        or whatever you think is the best way...
    """
    # `tf.audio.decode_wav` might be helpful
    # we want the audio stored in a way similar to  `scipy.io.wavfile.read(fn)`