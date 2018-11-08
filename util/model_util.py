from model import mlp
from model import rnn
import configparser

# def get_model(model_type, model_config_path, model_name):
def get_model(model_config_path):
    config = configparser.ConfigParser()
    config.read(model_config_path)
    model_type = config.get("Model", "type")
    print(model_type)

    if model_type == "MLP":
        model = mlp.MLP(config_file=model_config_path)
        return model

    if model_type == "RNN":
        model = rnn.RNN(config_file=model_config_path)
        return model
    raise ModelTypeNotFound()


class ModelTypeNotFound(Exception):
    def __init__(self):
        print("Model Type is not found")