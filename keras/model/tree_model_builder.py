import yaml
import tensorflow as tf
from keras.fn_util import get_keras_layer
from anytree import Node, RenderTree, LevelOrderGroupIter


class TreeModelBuilder:
    @staticmethod
    def __print_log(msg, debug=False):
        if debug:
            print(msg)

    def __init__(self, yaml_config_file):
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
        self._compile()

    def _build_layers(self):
        layers_config = self.yaml_config["config"]["layers"]
        for params in layers_config:
            layer = get_keras_layer(params)
            self.layer_dict[layer.name] = layer
            self.node_dict[layer.name] = Node(layer.name)
        # print(self.node_dict)
        for layer_name in self.layer_dict:
            self.__print_log(self.layer_dict[layer_name])
        self.__print_log("###############################################")

    def _build_tree(self):
        layers_config = self.yaml_config["config"]["layers"]
        for params in layers_config:
            parent_layer_name = params["config"]["name"]
            parent_node = self.node_dict[parent_layer_name]
            children_node = [self.node_dict[x] for x in params["inbound_layer"] if x is not None]
            parent_node.children = children_node
        self.__print_log("Tree is built")
        self.__print_log("###############################################")

    def _build_ffn(self):
        def _build_level(input_nodes):
            self.__print_log("Current Input Nodes")
            self.__print_log(input_nodes)

            for name in input_nodes:
                self.__print_log("Processing node " + name)
                self.__print_log("Number of child nodes: " + str(len(self.node_dict[name].children)))
                parent_layer = self.layer_dict[name]
                self.__print_log("Get parent layer")
                self.__print_log(type(parent_layer))

                if len(self.node_dict[name].children) == 1:
                    child_node = self.node_dict[name].children[0]
                    self.__print_log("Get child layer")
                    self.__print_log(type(self.layer_dict[child_node.name]))
                    if "InputLayer" in str(type(self.layer_dict[child_node.name])):
                        output_node = parent_layer(self.layer_dict[child_node.name].input)
                        self.inputs.append(self.layer_dict[child_node.name].input)
                    else:
                        output_node = parent_layer(self.layer_dict[child_node.name])
                elif len(self.node_dict[name].children) > 1:
                    concatenate_layer_list = []
                    for child_node in self.node_dict[name].children:
                        self.__print_log("Get child layer: " + child_node.name)
                        self.__print_log(type(self.layer_dict[child_node.name]))

                        if "InputLayer" in str(type(self.layer_dict[child_node.name])):
                            concatenate_layer_list.append(self.layer_dict[child_node.name].input)
                            self.inputs.append(self.layer_dict[child_node.name].input)
                        else:
                            concatenate_layer_list.append(self.layer_dict[child_node.name])
                    self.__print_log(concatenate_layer_list)
                    output_node = tf.keras.layers.concatenate(concatenate_layer_list)
                else:
                    output_node = parent_layer
                self.layer_dict[name] = output_node
                self.__print_log("***********************************************")

        root_node = None
        for node_name in self.node_dict:
            if self.node_dict[node_name].is_root:
                root_node = self.node_dict[node_name]
                self.root_node_name = node_name
                self.__print_log(RenderTree(self.node_dict[node_name]), debug=True)

        # for child_node in root_node.children:
        #     print(len(root_node.children))
        #     print(child_node.name)

        levels = [[node.name for node in children] for children in LevelOrderGroupIter(root_node)]
        levels.reverse()
        self.__print_log(levels)
        self.__print_log("###############################################")

        for level in levels:
            _build_level(level)
        self.__print_log("###############################################")

    def _build_model(self):
        self.outputs.append(self.layer_dict[self.root_node_name])
        self.__print_log(self.inputs)
        self.__print_log(self.outputs)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    def _compile(self):
        optimizer = self.yaml_config["config"]["optimizer"]
        loss = self.yaml_config["config"]["loss"]
        metrics = self.yaml_config["config"]["metrics"]
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def get_model(self):
        return self.model

