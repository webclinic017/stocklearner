import yaml
import tensorflow as tf
from feed import csv_data as stock_data
from keras.model.tree_model_builder import TreeModelBuilder


APP_CONFIG_FILE_PATH = "./tf_keras_sl_ops.yaml"


if __name__ == "__main__":
    # enable below when use keras and comment it when user estimator
    # tf.enable_eager_execution()

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

    builder = TreeModelBuilder(model_config_path)
    keras_model = builder.get_model()

    # train_dataset = stock_data.csv_input_fn_estimate(training_data_path, keras_model.input_names, one_hot=True)
    # train_dataset = train_dataset.batch(batch_size).repeat(repeat_time)

    # keras training
    # keras_model.fit(train_dataset, steps_per_epoch=5000, epochs=50)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)
    estimator.train(input_fn=lambda: stock_data.csv_input_fn_estimate(training_data_path,
                                                                      batch_size=batch_size,
                                                                      input_name=keras_model.input_names,
                                                                      one_hot=True),
                    steps=train_step)
