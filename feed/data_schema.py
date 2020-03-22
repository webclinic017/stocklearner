import os
from os.path import join

import tensorflow as tf
import yaml


class CSVDataSchema:
    def __init__(self, yaml_config_path):
        self.yaml_file = open(yaml_config_path, 'r', encoding='utf-8')
        self.yaml_config = yaml.load(self.yaml_file.read())

        self.output_size = self.yaml_config["output_size"]
        self.divided_by = self.yaml_config["divided_by"]
        self.is_rnn = self.yaml_config["is_rnn"]

        self.columns = []
        self.field_defaults = []
        for field in self.yaml_config["schema"]:
            self.columns.append(field["field_name"])
            self.field_defaults.append(field["default_value"])

    def get_input_fn(self):
        if self.yaml_config["train_method"] == "estimate":
            return self.tf_estimate_input_fn
        else:
            return self.tf_ds_input_fn

        raise NotImplementedError

    def tf_estimate_transformer_input_fn(self, csv_path, input_name, time_steps, batch_size, buffer_size=None,
                                         repeat=None, one_hot=False):
        def _parse_line(line):
            fields = tf.io.decode_csv(records=line, record_defaults=self.field_defaults)
            labels = fields[-1:]
            fields = fields[1:-1]
            features = tf.reshape(fields, [batch_size, time_steps, -1])

            if one_hot:
                labels = tf.add(labels, 11)
                labels = tf.divide(labels, self.divided_by)
                labels = tf.round(labels)
                labels = tf.cast(labels, dtype=tf.int64)
                labels = tf.one_hot(labels, self.output_size)
                labels = tf.reshape(labels, [-1, time_steps, self.output_size])
                # labels = tf.slice(labels, [0, time_steps - 1, 0], [batch_size, 1, self.output_size])
                # labels = tf.reshape(labels, [-1, self.output_size])
            else:
                labels = tf.reshape(labels, [batch_size, time_steps, -1])
                # labels = tf.slice(labels, [0, time_steps - 1, 0], [batch_size, 1, 1])
                # labels = tf.reshape(labels, [-1, 1])

            features = dict(zip(input_name, [features, labels]))
            print(features)
            print(labels)
            return features, labels

        filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store" and "_s" in f]
        print(filenames)
        dataset = tf.data.TextLineDataset(filenames).skip(0)

        dataset = dataset.batch(batch_size * time_steps, drop_remainder=True)
        dataset = dataset.map(_parse_line)

        if buffer_size is not None:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        if buffer_size is not None:
            dataset = dataset.repeat(repeat)

        return dataset

    def tf_estimate_input_fn(self, csv_path, input_name, time_steps, batch_size=None, buffer_size=None, repeat=None, one_hot=False):

        def _parse_line_rnn(line):
            fields = tf.io.decode_csv(records=line, record_defaults=self.field_defaults)
            labels = fields[-1:]
            fields = fields[1:-1]
            fields = tf.reshape(fields, [batch_size, time_steps, -1])
            features = dict(zip(input_name, [fields]))

            if one_hot:
                labels = tf.add(labels, 11)
                labels = tf.divide(labels, self.divided_by)
                labels = tf.round(labels)
                labels = tf.cast(labels, dtype=tf.int64)
                labels = tf.one_hot(labels, self.output_size)
                labels = tf.reshape(labels, [-1, time_steps, self.output_size])
                labels = tf.slice(labels, [0, time_steps - 1, 0], [batch_size, 1, self.output_size])
                labels = tf.reshape(labels, [-1, self.output_size])
            else:
                labels = tf.reshape(labels, [batch_size, time_steps, -1])
                labels = tf.slice(labels, [0, time_steps - 1, 0], [batch_size, 1, 1])
                labels = tf.reshape(labels, [-1, 1])
            return features, labels

        def _parse_line(line):
            fields = tf.io.decode_csv(records=line, record_defaults=self.field_defaults)
            labels = fields[-1:]
            fields = fields[1:-1]
            features = dict(zip(input_name, [fields]))
            if one_hot:
                labels = tf.add(labels, 11)
                labels = tf.divide(labels, self.divided_by)
                labels = tf.round(labels)
                labels = tf.cast(labels, dtype=tf.int64)
                labels = tf.one_hot(labels, self.output_size)
                print(labels)
                labels = tf.reshape(labels, [self.output_size])
                print(labels)
            return features, labels

        filenames = [join(csv_path, f) for f in os.listdir(csv_path) if f != ".DS_Store" and "_s" in f]
        dataset = tf.data.TextLineDataset(filenames).skip(0)

        if buffer_size is not None:
            dataset = dataset.shuffle(buffer_size=10000)

        if self.is_rnn:
            dataset = dataset.batch(batch_size * time_steps, drop_remainder=True)
            dataset = dataset.map(_parse_line_rnn)
        else:
            dataset = dataset.map(_parse_line)
            dataset = dataset.batch(batch_size)

        if buffer_size is not None:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        if buffer_size is not None:
            dataset = dataset.repeat(repeat)

        return dataset

    def tf_ds_input_fn(self, csv_path, batch_size=None, buffer_size=None, repeat=None, one_hot=False):
        def _parse_line(line):
            fields = tf.io.decode_csv(records=line, record_defaults=self.field_defaults)
            features = dict(zip(self.columns, fields))
            features.pop("DATE")
            label = features.pop("LABEL")

            if one_hot:
                label = tf.add(label, 11)
                label = tf.divide(label, self.divided_by)
                label = tf.round(label)
                label = tf.cast(label, dtype=tf.int64)
                label = tf.one_hot(label, self.output_size)
                label = tf.reshape(label, [self.output_size])

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
    tf.compat.v1.enable_eager_execution()

    data_schema_file_path = "../config_file/yaml_config/basic_data_schema.yaml"

    data_source = "../test_data/stock/"
    schema = CSVDataSchema(data_schema_file_path)
    # input_fn = schema.get_input_fn()
    # ds = input_fn(data_source, input_name="main_input", batch_size=3, time_steps=5)

    ds = schema.tf_estimate_transformer_input_fn(data_source,
                                                 input_name=["encoder_inputs", "decoder_inputs"],
                                                 time_steps=5,
                                                 batch_size=1,
                                                 one_hot=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    for i in range(5):
        next_xs, next_ys = iterator.get_next()
        print(next_xs)
        print(next_ys)
