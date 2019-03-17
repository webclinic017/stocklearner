import yaml
from keras.fn_util import get_keras_layer


class MLP:
    def __init__(self, yaml_config_file):
        yaml_file = open(yaml_config_file, 'r', encoding='utf-8')
        self.yaml_config = yaml.load(yaml_file.read())
        self.layer_dict = {}
        self._build_layers()

    def _build_layers(self):
        layers_config = self.yaml_config["config"]["layers"]
        for params in layers_config:
            layer = get_keras_layer(params)
            self.layer_dict[layer.name] = layer


if __name__ == "__main__":
    model = MLP("my_model.yaml")
    for layer_name in model.layer_dict:
        print(layer_name + ": " + str(type(model.layer_dict[layer_name])))
