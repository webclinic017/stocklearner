import os
import tensorflow as tf
from os.path import join

#           0       1       2       3        4      5         6               7           8           9
COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
BOUNDARIES = [-7, -5, -3, 0, 3, 5, 7]


def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    features.pop("DATE")
    label = features.pop("LABEL")
    return features, label


def csv_input_fn(csv_path, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
    # 2018-09-18 Remove .DS_Store for Mac OS
    filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store"]
    # print(filenames)
    dataset = tf.data.TextLineDataset(filenames).skip(0)
    if one_hot:
        pass
    else:
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
    ds = csv_input_fn(data_source, batch_size=32, repeat=-1)

    # ds = ds.batch(20)
    # ds = ds.repeat(-1)
    iterator = ds.make_one_shot_iterator()
    next_xs, next_ys = iterator.get_next()
    # print(next_xs)
    # print(next_ys)
    # with tf.Session() as sess:
    #     next_xs, next_ys = iterator.get_next()
    #     step = 0
    #     while step <= 50000:
    #         raw_xs, raw_ys = sess.run([next_xs, next_ys])
    #         print(raw_ys)
    #         step = step + 1
    x = tf.constant([0.34, 0.67, -1.34, 0.34, 0., 4.73, -3.23, -1.33, 0.34, 3.86, 2.19, 1.5,
                     0.21, 1.89, -0.83, 0., -2.5, -0.85, -4.55, -7.14, 2.56, -2.25, 1.28, -6.57,
                     -8.65, 2.46, -6.67, 2.57, -0.28, 0.56, 6.11, 10.99])

    # x = tf.constant([0, 1, 1, 3, 2, 3, 4])
    s = tf.shape(x)
    print(tf.reshape(x, [-1]))
    values, idx = tf.unique(tf.reshape(x, [-1]))
    n = tf.size(BOUNDARIES)
    a_1h_flat = tf.one_hot(idx, n)
    # Reshape to original shape
    a_1h = tf.reshape(a_1h_flat, tf.concat([s, [n]], axis=0))
    print(a_1h)
