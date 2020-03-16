

# setup
* install tensorflow (we used version 2.1.0)
* `pip install -r requirements.txt`

# get the data
download the TIMIT dataset from [this kaggle url]
(kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech) and 
unzip it into a directory named "data/timit" inside the root of this 
repository.

# run
`python noisy2txt.py`

# configuring the run
You can adjust options for the run by editing (or adding entries to) 
the `custom_static_params` dictionary at the bottom of "noisy2txt.py".

You can also run `stft_denoise.py` to train a larger denoisng model. A larger dataset need to be downloaded.

The model will be saved under the `model` directory and you can evaluate it with `sftf_eval.py`. 

# To listen to dataset as it will be fed to the model
`python timit_loader.py`
Note that this allows you to check that the transcripts are cropped 
properly and that the noise level is as desired.
