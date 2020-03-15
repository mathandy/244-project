import pandas as pd
from scipy.io import wavfile

fs, noise = wavfile.read("noise/AirConditioner_1.wav")
noise = noise.astype('float32')

def create_noisy_file(source_file, noise):
    ratio = 10
    for i in range(len(source_file["clean_file"])):
        clean_src = source_file.iloc[i, 0]
        noisy_dst = source_file.iloc[i, 2]
        fs, clean = wavfile.read(clean_src)
        clean = clean.astype('float32')
        noisy_file = (clean * (ratio - 1) // ratio + noise[:len(clean)] // ratio) / 32768
        wavfile.write(noisy_dst, fs, noisy_file)


source_file = pd.read_csv("data/train.csv")
create_noisy_file(source_file, noise)

create_noisy_file(pd.read_csv("data/test.csv"), noise)
