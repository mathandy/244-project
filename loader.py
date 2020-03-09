import os
import tensorflow as tf
from scipy.io import wavfile


def get_file_info(filename=os.path.join('data', 'train.csv')):
    """returns as list of: clean_file, transcript, noisy_file"""
    with open(filename, 'r') as f:
        file_info = []
        for i, line in enumerate(f.read().split('\n')):
            if not i:
                continue
            if not line.strip():
                continue
            clean_fp, transcript, noisy_fp = line.strip().split(',')
            file_info.append(
                (clean_fp.strip(), transcript.strip(), noisy_fp.strip())
            )
    return file_info


def loader():
    sample_list = get_file_info()
    samples = []
    for clean_fp, transcript, noisy_fp in sample_list:
        sr1, clean_wav = wavfile.read(clean_fp)
        sr2, noisy_wav = wavfile.read(noisy_fp)
        assert sr1 == 16000 and sr2 == 16000
        # yield clean_wav, noisy_wav, clean_fp, noisy_fp, transcript
        samples.append((clean_wav, noisy_wav, clean_fp, noisy_fp, transcript))
    return samples


def get_noise_filepaths(data_dir=os.path.join('data', 'noise')):
    noise_filepaths = []
    for fn in os.listdir(data_dir):
        if not fn.endswith('.wav'):
            continue
        noise_filepaths.append(os.path.join(data_dir, fn))
    return noise_filepaths


# def loader():
#     """Return a 6 (unshuffled) tf dataset objects:
#         clean_audio, clean_audio_filepath, noisy_audio, noisy_audio_filepath, noise type, label
#
#         or whatever you think is the best way...
#     """
#     # sample_list = get_file_info()
#     # ds = tf.data.Dataset.from_tensor_slices(sample_list)
#     ds = tf.data.Dataset.from_generator(
#         data_generator,
#         (tf.int16, tf.int16, tf.string, tf.string, tf.string)
#     )
#     ds = tf.data.Dataset.from_generator(data_generator, tf.int16)
#
#     @tf.function
#     def load_audio(sample_triple):
#         clean_fp, transcript, noisy_fp = sample_triple
#         clean_wav = tf.audio.decode_wav(clean_fp)
#         noisy_wav = tf.audio.decode_wav(noisy_fp)
#         return clean_wav, noisy_wav, clean_fp, transcript, noisy_fp
#
#     ds = ds.map(load_audio)
#     ds = ds.shuffle()
#     ds = ds.batch(32)
#     return ds


if __name__ == '__main__':
    ds = loader()
    from IPython import embed; embed()  ### DEBUG
