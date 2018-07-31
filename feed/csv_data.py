import os
import tensorflow as tf
from os.path import join

COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    features.pop("DATE")
    label = features.pop("LABEL")
    return features, label


def csv_input_fn(csv_path, batch_size=None, buffer_size=None, repeat=None):
    filenames = [join(csv_path, f) for f in os.listdir(csv_path)]
    dataset = tf.data.TextLineDataset(filenames).skip(1)
    dataset = dataset.map(_parse_line)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=10000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if buffer_size is not None:
        dataset = dataset.repeat(repeat)


    return dataset
