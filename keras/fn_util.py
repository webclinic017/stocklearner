from util.class_util import create_instance

TF_KERAS_LAYERS_MODULE = "tensorflow.keras.layers"


def get_keras_layer(config):
    class_name = config["class_name"]
    layer_config = config["config"]

    try:
        layer = create_instance(TF_KERAS_LAYERS_MODULE, class_name, **layer_config)
        return layer
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    my_config = {'class_name': 'InputLayer',
                 'config': {
                     'input_shape': (768, ),
                     'batch_size': 32,
                     'dtype': None,
                     'input_tensor': None,
                     'sparse': False,
                     'name': 'main_input'}
                 }
    dyn_layer = get_keras_layer(my_config)
    print(type(dyn_layer))
    print(dyn_layer.name)



