import yaml
import tensorflow as tf
from keras.fn_util import get_keras_layer
from anytree import Node, RenderTree, LevelOrderGroupIter


class TreeModelBuilder:
    def __init__(self, yaml_config_file):
        print(tf.executing_eagerly())
        yaml_file = open(yaml_config_file, 'r', encoding='utf-8')
        self.yaml_config = yaml.load(yaml_file.read())

        self.layer_dict = {}
        self.node_dict = {}

        self.inputs = []
        self.outputs = []

        self._build_layers()
        self._build_tree()
        self._build_ffn()
        self._build_model()

        self._manual_build_model()

    def _manual_build_model(self):
        main_input = tf.keras.layers.Input(shape=(None, 64), name='main_input')
        aux_input = tf.keras.layers.Input(shape=(None, 8), name='aux_input')

        dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')(main_input)
        concat = tf.keras.layers.concatenate([aux_input, dense1])
        dense2 = tf.keras.layers.Dense(32, activation='relu', name='dense2')(concat)
        dense3 = tf.keras.layers.Dense(16, activation='relu', name='dense3')(dense2)
        main_output = tf.keras.layers.Dense(8, activation='sigmoid', name='main_output')(dense3)

        self.x = tf.keras.Model(inputs=[main_input, aux_input], outputs=main_output)
        self.x.summary()

    def _build_layers(self):
        layers_config = self.yaml_config["config"]["layers"]
        for params in layers_config:
            layer = get_keras_layer(params)
            self.layer_dict[layer.name] = layer
            self.node_dict[layer.name] = Node(layer.name)
        # print(self.node_dict)
        for layer_name in self.layer_dict:
            print(self.layer_dict[layer_name])
        print("###############################################")

    def _build_tree(self):
        layers_config = self.yaml_config["config"]["layers"]
        for params in layers_config:
            parent_layer_name = params["config"]["name"]
            parent_node = self.node_dict[parent_layer_name]
            children_node = [self.node_dict[x] for x in params["inbound_layer"] if x is not None]
            parent_node.children = children_node
        print("Tree is built")
        print("###############################################")

    def _build_ffn(self):
        def _build_level(input_nodes):
            print("Current Input Nodes")
            print(input_nodes)

            for name in input_nodes:
                print("Processing node " + name)
                print("Number of child nodes: " + str(len(self.node_dict[name].children)))
                parent_layer = self.layer_dict[name]
                print("Get parent layer")
                print(type(parent_layer))

                if len(self.node_dict[name].children) == 1:
                    child_node = self.node_dict[name].children[0]
                    print("Get child layer")
                    print(type(self.layer_dict[child_node.name]))
                    if "InputLayer" in str(type(self.layer_dict[child_node.name])):
                        output_node = parent_layer(self.layer_dict[child_node.name].input)
                        self.inputs.append(self.layer_dict[child_node.name].input)
                    else:
                        output_node = parent_layer(self.layer_dict[child_node.name])
                elif len(self.node_dict[name].children) > 1:
                    concatenate_layer_list = []
                    for child_node in self.node_dict[name].children:
                        print("Get child layer: " + child_node.name)
                        print(type(self.layer_dict[child_node.name]))

                        if "InputLayer" in str(type(self.layer_dict[child_node.name])):
                            concatenate_layer_list.append(self.layer_dict[child_node.name].input)
                            self.inputs.append(self.layer_dict[child_node.name].input)
                        else:
                            concatenate_layer_list.append(self.layer_dict[child_node.name])
                    print(concatenate_layer_list)
                    output_node = tf.keras.layers.concatenate(concatenate_layer_list)
                else:
                    output_node = parent_layer
                self.layer_dict[name] = output_node
                print("***********************************************")

        root_node = None
        for node_name in self.node_dict:
            if self.node_dict[node_name].is_root:
                root_node = self.node_dict[node_name]
                self.root_node_name = node_name
                print(RenderTree(self.node_dict[node_name]))

        # for child_node in root_node.children:
        #     print(len(root_node.children))
        #     print(child_node.name)

        levels = [[node.name for node in children] for children in LevelOrderGroupIter(root_node)]
        levels.reverse()
        print(levels)
        print("###############################################")

        for level in levels:
            _build_level(level)
        print("###############################################")

    def _build_model(self):
        self.outputs.append(self.layer_dict[self.root_node_name])
        print(self.inputs)
        print(self.outputs)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()


if __name__ == "__main__":
    builder = TreeModelBuilder("my_model.yaml")
