

# setup
`pip install -r requirements.txt`

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
