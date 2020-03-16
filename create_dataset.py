from data_processing.clean_voice import CleanDataset
from data_processing.noise import NoiseDataset
from data_processing.dataset import Dataset
import warnings

warnings.filterwarnings(action='ignore')

clean_basepath = 'data/'
noise_basepath = 'data/'

cd = CleanDataset(clean_basepath, val_dataset_size=462)
clean_train_filenames, clean_val_filenames = cd.get_train_val_filenames()

nd = NoiseDataset(noise_basepath, val_dataset_size=4)
noise_train_filenames, noise_val_filenames = nd.get_train_val_filenames()

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=100)
print(1)
train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=1000)
print(2)
## Create Test Set
clean_test_filenames = cd.get_test_filenames()

noise_test_filenames = nd.get_test_filenames()
print(3)
print(noise_test_filenames)
test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
test_dataset.create_tf_record(prefix='test', subset_size=100, parallel=False)

