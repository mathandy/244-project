from loader2 import pad_or_trim
from loader import get_noise_filepaths

import os
from os.path import join as fpath
from random import choice
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import automatic_speech_recognition as asr


TIMIT_DATA_DIR = os.path.expanduser('~/244/data/timit/data')


def load(root_dir=TIMIT_DATA_DIR, subset='train'):
    data = []
    subset_dir = os.path.abspath(fpath(root_dir, subset.upper()))
    for dialect_id in os.listdir(subset_dir):
        dialect_dir = fpath(subset_dir, dialect_id)
        for speaker_id in os.listdir(dialect_dir):
            speaker_dir = fpath(dialect_dir, speaker_id)
            sentence_ids = \
                set([fn.split('.')[0] for fn in os.listdir(speaker_dir)])

            for sentence_id in sentence_ids:
                wav_fp = fpath(speaker_dir, sentence_id + '.WAV.wav')
                wrd_fp = fpath(speaker_dir, sentence_id + '.WRD')
                word_alignments = []
                for line in open(wrd_fp, 'r'):
                    if not line.strip():
                        continue
                    t0, t1, word = line.strip().split(' ')
                    word_alignments.append(
                        (int(t0.strip()), int(t1.strip()), word.strip()))
                data.append((wav_fp, word_alignments))
    return data


def load_and_crop_timit_to_length(subset="train", input_lenth=None):
    wav_filepaths, word_alignments = zip(*load(subset=subset))

    if input_lenth is None:
        input_lenth = np.inf

    # load audio and crop to just past last word before `input_length`
    audio_samples, transcripts = [], []
    for wav_fp, alignment in zip(wav_filepaths, word_alignments):
        sr, audio = wavfile.read(wav_fp)
        if len(audio) > input_lenth:
            transcript_words_to_include = []
            time_end = None
            previous_t1 = 0
            for t0, t1, w in alignment:
                if t1 < input_lenth:
                    transcript_words_to_include.append(w)
                    previous_t1 = t1
                else:
                    midpoint = int(round((previous_t1 + t0) / 2))
                    time_end = min(input_lenth, midpoint)
                    break
            if time_end is not None:
                audio = audio[:time_end]
        else:
            transcript_words_to_include = [w for _, _, w in alignment]

        audio_samples.append(pad_or_trim(audio, padded_length=input_lenth))
        transcripts.append(' '.join(transcript_words_to_include))

    return np.array(audio_samples), np.array(transcripts)


def load_timit_as_tf_dataset(subset="train", input_lenth=None, noisiness=0.5):
    audio, transcripts = \
        load_and_crop_timit_to_length(subset=subset, input_lenth=input_lenth)
    n_samples = len(audio)

    # create tensorflow dataset
    ds_clean = tf.data.Dataset.from_tensor_slices(audio.astype('float32'))

    noise_filepaths = get_noise_filepaths()

    # @tf.function
    def add_noise(clean_signal):
        """expects (and returns) 1D tensor"""
        audio_bytes = tf.io.read_file(choice(noise_filepaths))
        noise, sr = tf.audio.decode_wav(audio_bytes)
        noise = tf.squeeze(noise)
        clean_signal = tf.cast(clean_signal, tf.float32)

        # ensure noise is shorter than clean audio (trim if necessary)
        pad_length = tf.shape(clean_signal)[-1] - tf.shape(noise)[-1]
        if pad_length < 0:
            # randomly offset, then trim to length of clean_signal
            offset = tf.random.uniform((), 0, len(noise) - 1, tf.int32)
            noise = tf.roll(noise, offset, -1, name='pre-trim_noise_offset')
            noise = noise[:tf.shape(clean_signal)[0]]
        else:
            # pad_or_trim to length of clean_signal, then randomly offset
            noise = tf.pad(noise, [[0, pad_length]])
            offset = tf.random.uniform((), 0, len(noise) - 1, tf.int32)
            noise = tf.roll(noise, offset, -1, name='post-pad_noise_offset')

        # normalize
        clean_signal = clean_signal / tf.reduce_max(clean_signal)
        noise = noise / tf.reduce_max(noise)

        return (1 - noisiness) * clean_signal + noisiness * noise

    ds_noisy = ds_clean.map(add_noise)

    alphabet = asr.text.Alphabet(lang='en')
    encoded_transcripts = alphabet.get_batch_labels(transcripts)
    ds_encoded_transcripts = \
        tf.data.Dataset.from_tensor_slices(encoded_transcripts)
    return ds_clean, ds_noisy, ds_encoded_transcripts, n_samples


if __name__ == '__main__':
    from tempfile import gettempdir
    from util import renormalize_quantize_and_save
    tmpdir = fpath(gettempdir(), 'timit_loader_test')
    os.makedirs(tmpdir, exist_ok=True)

    dscl, dsns, dset, n_samples = load_timit_as_tf_dataset(
        subset="train", input_lenth=32000, noisiness=0.5)
    ds_zip = tf.data.Dataset.zip((dscl, dsns, dset))

    alphabet = asr.text.Alphabet(lang='en')

    for clean, noisy, enc_trans in ds_zip:
        transcript = alphabet.get_batch_transcripts([enc_trans.numpy()])
        print(transcript)

        # play clean
        tmp_clean_fp = fpath(tmpdir, 'tmp.wav')
        renormalize_quantize_and_save(clean.numpy(), tmp_clean_fp)
        os.system(f'play {tmp_clean_fp}')

        # play noisy
        tmp_noisy_fp = fpath(tmpdir, 'tmp.wav')
        renormalize_quantize_and_save(noisy.numpy(), tmp_noisy_fp)
        os.system(f'play {tmp_noisy_fp}')

        input("Press enter to continue.")
