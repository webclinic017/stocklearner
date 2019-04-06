import os
import tensorflow as tf
from os.path import join

#           0       1       2       3        4      5         6               7           8           9
COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
BOUNDARIES = [-7, -5, -3, 0, 3, 5, 7]

OUTPUT_SIZE = 8
DIVIDE_BY = 3


def csv_input_fn_estimate_rnn(csv_path, input_name, time_steps, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
    def _parse_line(line):
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
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
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
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
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        features = dict(zip(COLUMNS, fields))
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
    tf.enable_eager_execution()

    data_source = "../test_data/stock/"
    ds = csv_input_fn(data_source, batch_size=32, repeat=-1, one_hot=True)
    rnn_ds = csv_input_fn_estimate_rnn(data_source, input_name="main_input", batch_size=3, time_steps=5)

    # ds = ds.batch(20)
    # ds = ds.repeat(-1)
    iterator = rnn_ds.make_one_shot_iterator()
    next_xs, next_ys = iterator.get_next()
    print(next_xs)
    print(next_ys)
    s = tf.slice(next_ys, [0, 4, 0], [3, 1, 1])
    print(s)
    s = tf.reshape(s, [-1, 1])
    print(s)
    # with tf.Session() as sess:
    #     next_xs, next_ys = iterator.get_next()
    #     step = 0
    #     while step <= 50000:
    #         raw_    #         step = step + 1xs, raw_ys = sess.run([next_xs, next_ys])
    #         print(raw_ys)
