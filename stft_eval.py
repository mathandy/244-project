from data_processing.feature_extractor import FeatureExtractor
import numpy as np
from stft_model import build_model, simple_denoiser
from scipy.io import wavfile
import librosa
import pandas as pd
import os

windowLength = 256
overlap = round(0.25 * windowLength)  # overlap of 75%
ffTLength = windowLength
inputFs = 48e3
fs = 16e3
numFeatures = ffTLength//2 + 1
numSegments = 8


def prepare_input_features(stft_features):
    # Phase Aware Scaling: To avoid extreme differences (more than
    # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments


def get_denoised_audio(noisyAudio):
    noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    # Paper: Besides, spectral phase was not used in the training phase.
    # At reconstruction, noisy spectral phase was used instead to
    # perform in- verse STFT and recover human speech.
    noisyPhase = np.angle(noise_stft_features)
    noise_stft_features = np.abs(noise_stft_features)

    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std
    predictors = prepare_input_features(noise_stft_features)
    predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
    STFTFullyConvolutional = model.predict(predictors)
    denoisedAudioFullyConvolutional = revert_features_to_audio(STFTFullyConvolutional,
                                                               noisyPhase, noiseAudioFeatureExtractor, mean, std)
    return denoisedAudioFullyConvolutional


def revert_features_to_audio(features, phase, noiseAudioFeatureExtractor, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    # features = librosa.db_to_power(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)


def read_audio(filepath, sample_rate, normalize=True):
    # print(f"Reading: {filepath}").
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
    return audio, sr


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio


def l2_norm(vector):
    return np.square(vector)


def SDR(denoised, cleaned, eps=1e-7): # Signal to Distortion Ratio
    a = l2_norm(denoised)
    b = l2_norm(denoised - cleaned[:len(denoised)])
    a_b = a / (b + eps)
    return np.mean(10 * np.log10(a_b + eps))


if __name__ == '__main__':
    # model = build_model(l2_strength=0.0, numFeatures=numFeatures, numSegments=numSegments, learning_rate=0.01, beta_1=0.5, beta_2=0.3, epsilon=0)
    model = simple_denoiser(numFeatures=numFeatures, numSegments=numSegments)
    model.load_weights('models/denoiser_cnn_log_mel_generator.h5')
    test = pd.read_csv("data/test.csv")
    noise_audios = pd.read_csv("data/test_noise.csv")["noise"]
    transcripts = test['transcript']
    clean_audios = test['clean_file']
    denoised_files = pd.DataFrame(data={"transcript": transcripts})
    for noise in noise_audios:
        noise_audio, sr = read_audio(noise, 16000)
        noise_name = noise.split("/")[-1].split(".")[0]
        sdr = 0
        # TODO: change the noisy here if storing denoised files
        directory_to_store = os.path.join("simple_denoised", noise_name)
        if not os.path.isdir(directory_to_store):
            os.makedirs(directory_to_store)
        denoised_audios = []
        for i in range(len(clean_audios)):
            clean_audio, sr = read_audio(clean_audios[i], 16000)
            noisy_audio = add_noise_to_clean_audio(clean_audio, noise_audio)
            denoised = get_denoised_audio(noisy_audio)
            sdr += SDR(denoised, clean_audio)
            clean_filename = clean_audios[i].split("/")[-1]
            denoised_audio_path = os.path.join(directory_to_store, clean_filename)
            wavfile.write(denoised_audio_path, 16000, denoised)
            denoised_audios.append(denoised_audio_path)
        denoised_files[noise_name] = denoised_audios
        print(f"{noise_name}:  {sdr / len(clean_audios)}")
    denoised_files.to_csv("simple_denoised/denoised_files.csv", index=False)





