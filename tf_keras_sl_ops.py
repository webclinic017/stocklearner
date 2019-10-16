import yaml
import os
import tensorflow as tf
from feed import csv_data as stock_data
from feed.data_schema import CSVDataSchema
from keras.model.tree_model_builder import TreeModelBuilder
from keras.callbacks import SavedModelCallback


APP_CONFIG_FILE_PATH = "./tf_keras_sl_ops_rnn.yaml"

# enable below when use keras and comment it when user estimator
# tf.enable_eager_execution()

yaml_file = open(APP_CONFIG_FILE_PATH, 'r', encoding='utf-8')
yaml_config = yaml.load(yaml_file.read())

if "rnn" in APP_CONFIG_FILE_PATH:
    time_steps = yaml_config["dataset"]["time_steps"]

schema_config_path = yaml_config["dataset"]["schema_path"]
training_data_path = yaml_config["dataset"]["training_data_path"]
eval_data_path = yaml_config["dataset"]["eval_data_path"]
batch_size = yaml_config["dataset"]["batch_size"]
repeat_time = yaml_config["dataset"]["repeat_time"]

model_config_path = yaml_config["model"]["config_path"]
model_dir = yaml_config["model"]["output_dir"]
log_dir = yaml_config["model"]["log_dir"]

train_step = yaml_config["run"]["train"]["epochs"]
steps_per_epoch = yaml_config["run"]["train"]["steps_per_epoch"]
eval_step = yaml_config["run"]["eval"]["epochs"]
train_use = yaml_config["run"]["use"]
run_for = yaml_config["run"]["for"]


def train():
    builder = TreeModelBuilder(model_config_path)
    keras_model = builder.get_model()

    if train_use == "keras":
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size)
        savedmodel_callback = SavedModelCallback(model_dir)

        # MLP
        # train_dataset = stock_data.csv_input_fn_estimate(training_data_path, keras_model.input_names, one_hot=True)
        # train_dataset = train_dataset.batch(batch_size).repeat(repeat_time)

        schema = CSVDataSchema(schema_config_path)
        input_fn = schema.get_input_fn()
        train_dataset = input_fn(training_data_path,
                                 keras_model.input_names,
                                 time_steps,
                                 batch_size,
                                 one_hot=True)

        # RNN
        # train_dataset = stock_data.csv_input_fn_estimate_rnn(training_data_path,
        #                                                      keras_model.input_names,
        #                                                      time_steps, batch_size,
        #                                                      one_hot=True)

        train_dataset = train_dataset.repeat(-1)
        # keras training
        keras_model.fit(train_dataset,
                        steps_per_epoch=steps_per_epoch,
                        epochs=train_step,
                        callbacks=[tensorboard_callback, savedmodel_callback])
    else:
        # TODO:
        # run_config = cl.create_instance("tf.estimator", yaml_config["run_config"])

        estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)
        # MLP
        # estimator.train(input_fn=lambda: stock_data.csv_input_fn_estimate(training_data_path,
        #                                                                   batch_size=batch_size,
        #                                                                   input_name=keras_model.input_names,
        #                                                                   one_hot=True),
        #                 steps=train_step)

        # RNN
        estimator.train(input_fn=lambda: stock_data.csv_input_fn_estimate_rnn(training_data_path,
                                                                              input_name=keras_model.input_names,
                                                                              batch_size=batch_size,
                                                                              time_steps=time_steps,
                                                                              one_hot=True))


def evaluate():
    if not os.path.exists(model_dir):
        raise RuntimeError("Model is not trained")

    if train_use == "keras":
        if tf.__version__ == "1.13.1":
            keras_model = tf.contrib.saved_model.load_keras_model(model_dir)
        else:
            keras_model = tf.keras.experimental.load_from_saved_model(model_dir)

        keras_model.summary()

        schema = CSVDataSchema(schema_config_path)
        input_fn = schema.get_input_fn()
        eval_dataset = input_fn(eval_data_path, keras_model.input_names, time_steps, batch_size, one_hot=True)
        keras_model.evaluate(eval_dataset, batch_size=batch_size, steps=eval_step)
    else:
        # TODO: evaluate by using tf.estimator
        pass


def predict(inputs):
    # TODO: predict with inputs
    pass


def main():
    if run_for == "train":
        train()
    elif run_for == "evaluate":
        evaluate()
    elif run_for == "predict":
        predict(None)


if __name__ == "__main__":
    main()

