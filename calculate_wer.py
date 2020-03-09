import numpy as np
import automatic_speech_recognition as asr
from collections import namedtuple
import pandas as pd


Metric = namedtuple('Metric', ['transcript', 'prediction', 'wer', 'cer'])


def get_metrics(sources, destinations):
    """ Calculate base metrics in one batch: WER and CER. """
    for source, destination in zip(sources, destinations):
        wer_distance, *_ = asr.evaluate.distance.edit_distance(source.split(),
                                                               destination.split())
        wer = wer_distance / len(destination.split())

        cer_distance, *_ = asr.evaluate.distance.edit_distance(list(source),
                                                               list(destination))
        cer = cer_distance / len(destination)
        yield Metric(destination, source, wer, cer)


pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.summary()     # TensorFlow model

test_dataset = asr.dataset.Audio.from_csv('noisy/noisy_files.csv', batch_size=64)
metrics = []
i = 0
for data, transcripts in test_dataset:
    predictions = pipeline.predict(data)
    batch_metrics = get_metrics(predictions, transcripts)
    metrics.extend(batch_metrics)
    i=i+1
    print(i)
metrics = pd.DataFrame(metrics)
print(f'WER: {metrics.wer.mean()}    CER: {metrics.cer.mean()}')

