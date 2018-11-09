import os
import tensorflow as tf
from os.path import join

#           0       1       2       3        4      5         6               7           8           9
COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]


def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    features.pop("DATE")
    label = features.pop("LABEL")
    return features, label


def csv_input_fn(csv_path, batch_size=None, buffer_size=None, repeat=None):
    # 2018-09-18 Remove .DS_Store for Mac OS
    filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store"]
    # print(filenames)
    dataset = tf.data.TextLineDataset(filenames).skip(1)
    dataset = dataset.map(_parse_line)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=10000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    if buffer_size is not None:
        dataset = dataset.repeat(repeat)

    return dataset


# if __name__ == "__main__":
#     data_source = "/Users/alex/Desktop/output/000004"
#     dataset = csv_input_fn(data_source)
#
#     dataset = dataset.batch(20)
#     dataset = dataset.repeat(-1)
#     iterator = dataset.make_one_shot_iterator()
#
#     with tf.Session() as sess:
#         next_xs, next_ys = iterator.get_next()
#         step = 0
#         while step <= 50000:
#             raw_xs, raw_ys = sess.run([next_xs, next_ys])
#             print(raw_ys)
#             step = step + 1
