import tensorflow as tf

# from keras.model.layer.attention_layer import SelfAttention
from keras.model.layer.ffn_layer import FeedFowardNetwork

APP_CONFIG_FILE_PATH = "./tf_keras_sl_ops_rnn.yaml"

tf.enable_eager_execution()


class TBuilder:
    def __init__(self, params):
        self.params = params
        # input shape -> [time_step, feature_length]
        self.encode_inputs = tf.keras.layers.Input(shape=[params["time_steps"], 8], batch_size=params["batch_size"], name="encode_inputs")
        self.decode_inputs = tf.keras.layers.Input(shape=[params["time_steps"], 8], batch_size=params["batch_size"], name="decode_inputs")
        # self.decode_inputs = tf.keras.layers.Input(batch_shape=(params["batch_size"], 8), name="decode_inputs")

        self.params["attention_bias"] = get_attention_bias(self.encode_inputs)

        print("TBuilder->__init__:self.encode_inputs")
        print(self.encode_inputs)

        print("TBuilder->__init__:self.target_inputs")
        print(self.decode_inputs)

        self.encode_stack = EncodeStack(params)
        self.decode_stack = DecodeStack(params)

        self.encode_outputs = self.encode(self.encode_inputs)
        print("TBuilder->__init__:self.encode passed")
        print("#############################################")
        print("TBuilder->__init__:self.encode_output")
        print(self.encode_outputs)
        self.decode_output = self.decode(self.decode_inputs, self.encode_outputs)
        print("TBuilder->__init__:self.decode passed")
        self.model = tf.keras.Model(inputs=[self.encode_inputs, self.decode_inputs], outputs=self.decode_output)

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def get_model(self):
        return self.model

    def encode(self, inputs):
        inputs = tf.keras.layers.Dense(self.params["hidden_size"])(inputs)
        print("TBuilder->encoder:inputs")
        print(inputs)
        return self.encode_stack(inputs)

    def decode(self, target_inputs, encoder_outputs):
        print("TBuilder->decoder:target_inputs")
        print(target_inputs)

        print("TBuilder->decoder:encoder_outputs")
        print(encoder_outputs)

        inputs = tf.keras.layers.Dense(self.params["hidden_size"])(target_inputs)

        print("TBuilder->decoder:inputs")
        print(inputs)

        # Run values
        length = tf.shape(inputs)[1]
        decoder_self_attention_bias = get_decoder_self_attention_bias(length)

        print("TBuilder->decoder:decoder_self_attention_bias")
        print(decoder_self_attention_bias)
        outputs = self.decode_stack(inputs,
                                    encoder_outputs=encoder_outputs,
                                    decoder_self_attention_bias=decoder_self_attention_bias)
        print("#############################################")
        logits = tf.keras.layers.Dense(target_inputs.shape[-1], activation="softmax")(outputs)
        return logits
        # return outputs


class DecodeStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(DecodeStack, self).__init__()
        self.params = params
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], params['train'])
            enc_dec_attention_layer = Attention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], params['train'])
            feed_forward_network = FeedFowardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], params['train'], params["allow_ffn_pad"])

        self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, params),
            PrePostProcessingWrapper(enc_dec_attention_layer, params),
            PrePostProcessingWrapper(feed_forward_network, params)])

        self.output_normalization = LayerNormalization(params["hidden_size"])

    # def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, cache=None):
    def call(self, inputs, **kwargs):
        # decoder_attention_bias = get_attention_bias(encoder_outputs)
        decoder_inputs = inputs
        attention_bias = self.params["attention_bias"]
        encoder_outputs = kwargs["encoder_outputs"]
        decoder_self_attention_bias = kwargs["decoder_self_attention_bias"]
        cache = None

        print("DecodeStack->call:decoder_inputs")
        print(decoder_inputs)

        print("DecodeStack->call:encoder_outputs")
        print(encoder_outputs)

        print("DecodeStack->call:attention_bias")
        # print(attention_bias)

        print("DecodeStack->call:decoder_self_attention_bias")
        print(decoder_self_attention_bias)

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, bias=decoder_self_attention_bias, cache=layer_cache)
                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, y=encoder_outputs, bias=attention_bias)
                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)


class EncodeStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncodeStack, self).__init__()
        self.params = params
        self.layers = []

        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], params['train'])
            feed_forward_network = FeedFowardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], params['train'], params["allow_ffn_pad"])

            self.layers.append([PrePostProcessingWrapper(self_attention_layer, params),
                                PrePostProcessingWrapper(feed_forward_network, params)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, inputs):
        print("EncodeStack->call:inputs")
        print(inputs)
        # attention_bias = get_attention_bias(inputs)
        attention_bias = self.params["attention_bias"]
        print("EncodeStack->call:attention_bias")
        # print(attention_bias)
        # inputs_padding = None

        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    inputs = self_attention_layer(inputs, bias=attention_bias)
                    # inputs = self_attention_layer(inputs)
                with tf.variable_scope("ffn"):
                    inputs = feed_forward_network(inputs)

        return self.output_normalization(inputs)


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        self.train = params["train"]

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def call(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.

        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        print("Attention->call:logits")
        print(logits)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


def get_attention_bias(inputs):
    print("get_attention_bias:inputs")
    print(inputs.shape)
    batch_size = inputs.shape[0]
    time_step = inputs.shape[1]

    attention_bias = tf.get_variable("attention_bias",
                                     shape=[batch_size, time_step],
                                     initializer=tf.initializers.constant(value=0))
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.

    Returns:
      float tensor of shape [1, 1, length, length]
    """

    _NEG_INF = -1e9
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


if __name__ == "__main__":
    tf.enable_eager_execution()

    # [time_steps, feature_length]
    p = dict()
    p["log_dir"] = "./logs"
    p["model_dir"] = "./output/"
    p["batch_size"] = 128
    p["time_steps"] = 5
    p["input_length"] = 10
    p["vocab_size"] = 33708
    p["num_hidden_layers"] = 5
    p["num_heads"] = 128
    p["hidden_size"] = 1024
    p["attention_dropout"] = 0.5
    p["relu_dropout"] = 0.5
    p["filter_size"] = 2048
    p["allow_ffn_pad"] = True
    p["layer_postprocess_dropout"] = 0.5
    p["train"] = True
    p["initializer_gain"] = 1.0

    b = TBuilder(p)
    keras_model = b.get_model()

    # from keras.callbacks import SavedModelCallback
    from feed.data_schema import CSVDataSchema

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=p["log_dir"], batch_size=p["batch_size"])
    csv_callback = tf.keras.callbacks.CSVLogger(filename="./logs/training.log")
    # savedmodel_callback = SavedModelCallback(p["model_dir"])

    schema = CSVDataSchema("./config_file/yaml_config/basic_data_schema.yaml")
    # input_fn = schema.get_input_fn()

    train_dataset = schema.tf_estimate_transformer_input_fn("/home/a1exff/Output/Train",
                                                            keras_model.input_names,
                                                            p["time_steps"],
                                                            p["batch_size"],
                                                            one_hot=True)

    train_dataset = train_dataset.repeat(50000)

    # keras training
    keras_model.fit(train_dataset,
                    steps_per_epoch=5000,
                    epochs=500,
                    callbacks=[tensorboard_callback, csv_callback])
