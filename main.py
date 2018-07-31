import os
import configparser
import tensorflow as tf
from util import dl_util
from os.path import join

CONFIG_FILE_NAME = "app.config"

config = configparser.ConfigParser()
config.read(join(os.getcwd(), CONFIG_FILE_NAME))
data_feed_path = config.get("Application", "data_feed_path")
dnn_model = config.get("Application", "dnn_model")
dnn_type = config.get("Application", "dnn_type")
dnn_config_path = join(join(os.getcwd(), "config"), dnn_model + "." + dnn_type)

batch_size = 10

flags = tf.app.flags
flags.DEFINE_string("data_dir", "D:\\Output", "Directory for stock data")
FLAGS = flags.FLAGS

COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]
FIELD_DEFAULTS = [["null"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
BOUNDARIES = [-7, -5, -2, 0, 2, 5, 7]


def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    features.pop("DATE")
    label = features.pop("LABEL")
    return features, label


def main(argv):
    print(dnn_config_path)
    print(FLAGS.data_dir)

    if FLAGS.data_dir is None:
        print("!!!!!")
        return
    else:
        filenames = [join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir)]
        dataset = tf.data.TextLineDataset(filenames).skip(1)
        dataset = dataset.map(_parse_line)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(20)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_example, next_label = iterator.get_next()

        with tf.Session() as sess:
            for i in range(1):
                features, labels = sess.run([next_example, next_label])
                print(features)
                print("**********************************************")
                print(type(labels))


if __name__ == "__main__":
    tf.app.run()