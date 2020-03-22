import os
from os.path import join

import tensorflow as tf

#           0       1       2       3        4      5         6               7           8           9
BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
BASIC_FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
BOUNDARIES = [-7, -5, -3, 0, 3, 5, 7]

OUTPUT_SIZE = 8
DIVIDE_BY = 3


def csv_input_fn_estimate_rnn(csv_path, input_name, time_steps, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
    def _parse_line(line):
        fields = tf.io.decode_csv(records=line, record_defaults=BASIC_FIELD_DEFAULTS)
        labels = fields[-1:]
        fields = fields[1:-1]
        fields = tf.reshape(fields, [batch_size, time_steps, -1])
        features = dict(zip(input_name, [fields]))

        if one_hot:
            labels = tf.add(labels, 11)
            labels = tf.divide(labels, DIVIDE_BY)
            labels = tf.round(labels)
            labels = tf.cast(labels, dtype=tf.int64)
            labels = tf.one_hot(labels, OUTPUT_SIZE)
            labels = tf.reshape(labels, [-1, time_steps, OUTPUT_SIZE])
            labels = tf.slice(labels, [0, time_steps-1, 0], [batch_size, 1, OUTPUT_SIZE])
            labels = tf.reshape(labels, [-1, OUTPUT_SIZE])
        else:
            labels = tf.reshape(labels, [batch_size, time_steps, -1])
            labels = tf.slice(labels, [0, time_steps - 1, 0], [batch_size, 1, 1])
            labels = tf.reshape(labels, [-1, 1])
        return features, labels

    filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store" and "_s" in f]
    dataset = tf.data.TextLineDataset(filenames).skip(0)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size * time_steps, drop_remainder=True)
    dataset = dataset.map(_parse_line)

    if buffer_size is not None:
        dataset = dataset.repeat(repeat)

    return dataset


def csv_input_fn_estimate(csv_path, input_name, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
    def _parse_line(line):
        fields = tf.io.decode_csv(records=line, record_defaults=BASIC_FIELD_DEFAULTS)
        labels = fields[-1:]
        fields = fields[1:-1]
        features = dict(zip(input_name, [fields]))
        if one_hot:
            labels = tf.add(labels, 11)
            labels = tf.divide(labels, DIVIDE_BY)
            labels = tf.round(labels)
            labels = tf.cast(labels, dtype=tf.int64)
            labels = tf.one_hot(labels, OUTPUT_SIZE)
            print(labels)
            labels = tf.reshape(labels, [OUTPUT_SIZE])
            print(labels)
        return features, labels

    filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store" and "_s" in f]
    dataset = tf.data.TextLineDataset(filenames).skip(0)
    dataset = dataset.map(_parse_line)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=10000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if buffer_size is not None:
        dataset = dataset.repeat(repeat)

    return dataset


def csv_input_fn(csv_path, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
    def _parse_line(line):
        fields = tf.io.decode_csv(records=line, record_defaults=BASIC_FIELD_DEFAULTS)
        features = dict(zip(BASIC_COLUMNS, fields))
        features.pop("DATE")
        label = features.pop("LABEL")

        if one_hot:
            label = tf.add(label, 11)
            label = tf.divide(label, DIVIDE_BY)
            label = tf.round(label)
            label = tf.cast(label, dtype=tf.int64)
            label = tf.one_hot(label, OUTPUT_SIZE)
            label = tf.reshape(label, [OUTPUT_SIZE])

        return features, label

    # 2018-09-18 Remove .DS_Store for Mac OS
    filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store"]
    # print(filenames)
    dataset = tf.data.TextLineDataset(filenames).skip(0)
    dataset = dataset.map(_parse_line)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=10000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if buffer_size is not None:
        dataset = dataset.repeat(repeat)

    return dataset


if __name__ == "__main__":
    type = "basic"
    if type == "basic":
        data_source = "../test_data/stock/basic/"
        schema_config_path = "../config_file/schema/basic_data_schema.yaml"
    else:
        data_source = "../test_data/stock/tech/"
        schema_config_path = "../config_file/schema/tech_data_schema.yaml"

    p_time_steps = 10
    p_batch_size = 32
    p_input_names = ["main_input"]

    from feed.data_schema import CSVDataSchema

    schema = CSVDataSchema(schema_config_path)
    input_fn = schema.get_input_fn()
    train_dataset = input_fn(data_source,
                             p_input_names,
                             p_time_steps,
                             p_batch_size,
                             one_hot=True)

    train_dataset = train_dataset.repeat(-1)

    iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    next_xs, next_ys = iterator.get_next()
    print(next_xs)  # [batch_size, time_steps, features]
    print(next_ys)  # [batch_size, one_hot_label]
