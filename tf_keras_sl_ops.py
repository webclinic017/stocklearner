import yaml
import tensorflow as tf
import numpy as np
from feed import csv_data as stock_data
from util import dl_util
# from util import class_util as cl
from keras.model.tree_model_builder import TreeModelBuilder


APP_CONFIG_FILE_PATH = "./tf_keras_sl_ops.yaml"

if __name__ == "__main__":
    tf.enable_eager_execution()

    yaml_file = open(APP_CONFIG_FILE_PATH, 'r', encoding='utf-8')
    yaml_config = yaml.load(yaml_file.read())

    training_data_path = yaml_config["dataset"]["training_data_path"]
    eval_data_path = yaml_config["dataset"]["eval_data_path"]
    batch_size = yaml_config["dataset"]["batch_size"]
    repeat_time = yaml_config["dataset"]["repeat_time"]

    model_config_path = yaml_config["model"]["config_path"]
    model_dir = yaml_config["model"]["output_dir"]

    train_step = yaml_config["train"]["steps"]

    # run_config = cl.create_instance("tf.estimator", yaml_config["run_config"])

    train_dataset = stock_data.csv_input_fn(training_data_path)
    # eval_dataset = stock_data.csv_input_fn(eval_data_path)
    print(train_dataset.output_shapes)
    train_dataset = train_dataset.batch(batch_size).repeat(repeat_time)

    iterator = train_dataset.make_one_shot_iterator()
    feature, label = iterator.get_next()
    print(feature)
    print(label)

    builder = TreeModelBuilder(model_config_path)
    model = builder.get_model()
    #
    # estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
    #                                                   model_dir=model_dir)

    # estimator.train(input_fn=lambda:stock_data.csv_input_fn(training_data_path))
    for i in range(train_step):
        raw_xs, raw_ys = iterator.get_next()
        batch_xs = np.array(dl_util.dict_to_list(raw_xs))
        batch_ys = np.array(dl_util.one_hot(raw_ys.numpy()))
        model.fit(x=batch_xs, y=batch_ys, batch_size=batch_size)

