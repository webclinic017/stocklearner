from __future__ import absolute_import, division, print_function, unicode_literals

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from feed.data_schema import CSVDataSchema

p_num_layers = 4
p_num_heads = 8
p_d_model = 512
p_dff = 2048  # hidden_size
p_dropout_rate = 0.1
p_time_steps = 10
p_output_size = 8
p_batch_size = 128
p_input_names = ["main_input"]
EPOCHS = 10

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

data_type = "tech"
if data_type == "basic":
    data_source = "../../test_data/stock/basic/"
    schema_config_path = "../../config_file/schema/basic_data_schema.yaml"
else:
    data_source = "../../test_data/stock/tech/"
    schema_config_path = "../../config_file/schema/tech_data_schema.yaml"

schema = CSVDataSchema(schema_config_path)
train_dataset = schema.tf_estimate_transformer_input_fn(data_source,
                                                        p_input_names,
                                                        p_time_steps,
                                                        p_batch_size,
                                                        one_hot=True)

train_dataset = train_dataset.repeat(100)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    print("positional_encoding->position")
    print(position)
    print("positional_encoding->d_model")
    print(d_model)
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    print("positional_encoding->angle_rads")
    print(angle_rads.shape)
    pos_encoding = angle_rads[np.newaxis, ...]
    print("positional_encoding->pos_encoding")
    print(pos_encoding.shape)
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    print("create_padding_mask->seq")
    print(seq.shape)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    print("create_padding_mask->return")
    print(seq[:, tf.newaxis, tf.newaxis, :].shape)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    print("scaled_dot_product_attention->q")
    print(q.shape)
    print("scaled_dot_product_attention->k")
    print(k.shape)
    print("scaled_dot_product_attention->v")
    print(v.shape)
    if mask is not None:
        print("scaled_dot_product_attention->mask")
        print(mask.shape)
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    print("scaled_dot_product_attention->matmul_qk")
    print(matmul_qk.shape)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    print("scaled_dot_product_attention->dk")
    print(dk.shape)

    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    print("scaled_dot_product_attention->scaled_attention_logits")
    print(scaled_attention_logits.shape)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        print("scaled_dot_product_attention->scaled_attention_logits after add")
        print(scaled_attention_logits.shape)
        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    print("scaled_dot_product_attention->output")
    print(output.shape)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        print("MultiHeadAttention.split_heads->x")
        print(x.shape)
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        print("MultiHeadAttention.split_heads->x after reshape")
        print(x.shape)
        print("MultiHeadAttention.split_heads->x after transpose")
        print(tf.transpose(x, perm=[0, 2, 1, 3]).shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        print("MultiHeadAttention.call->v")
        print(v.shape)
        print("MultiHeadAttention.call->k")
        print(k.shape)
        print("MultiHeadAttention.call->q")
        print(q.shape)
        if mask is not None:
            print("MultiHeadAttention.call->mask")
            print(mask.shape)

        batch_size = tf.shape(q)[0]
        print("MultiHeadAttention.call->batch_size")
        print(batch_size.shape)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        print("MultiHeadAttention.call->v after wv")
        print(v.shape)
        print("MultiHeadAttention.call->k after wk")
        print(k.shape)
        print("MultiHeadAttention.call->q after wq")
        print(q.shape)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        print("MultiHeadAttention.call->v after split_heads")
        print(v.shape)
        print("MultiHeadAttention.call->k after split_heads")
        print(k.shape)
        print("MultiHeadAttention.call->q after split_heads")
        print(q.shape)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        print("MultiHeadAttention.call->scaled_attention")
        print(scaled_attention.shape)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        print("MultiHeadAttention.call->scaled_attention after transpose")
        print(scaled_attention.shape)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        print("MultiHeadAttention.call->concat_attention")
        print(concat_attention.shape)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        print("MultiHeadAttention.call->output")
        print(output.shape)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        print("EncoderLayer.call->x")
        print(x.shape)
        if mask is not None:
            print("EncoderLayer.call->mask")
            print(mask.shape)
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        print("EncoderLayer.call->attn_output after mha")
        print(attn_output.shape)
        attn_output = self.dropout1(attn_output, training=training)
        print("EncoderLayer.call->attn_output after dropout")
        print(attn_output.shape)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        print("EncoderLayer.call->out1 after layer norm")
        print(out1.shape)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        print("EncoderLayer.call->ffn_output")
        print(ffn_output.shape)
        ffn_output = self.dropout2(ffn_output, training=training)
        print("EncoderLayer.call->ffn_output after dropout")
        print(ffn_output.shape)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        print("EncoderLayer.call->out2 after layer norm")
        print(out2.shape)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        print("DecoderLayer.call->x")
        print(x.shape)
        print("DecoderLayer.call->enc_output")
        print(enc_output.shape)
        if look_ahead_mask is not None:
            print("DecoderLayer.call->look_ahead_mask")
            print(look_ahead_mask.shape)
        if padding_mask is not None:
            print("DecoderLayer.call->padding_mask")
            print(padding_mask.shape)

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        print("DecoderLayer.call->attn1")
        print(attn1.shape)

        attn1 = self.dropout1(attn1, training=training)
        print("DecoderLayer.call->attn1 after dropout")
        print(attn1.shape)

        out1 = self.layernorm1(attn1 + x)
        print("DecoderLayer.call->out1")
        print(out1.shape)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        print("DecoderLayer.call->attn2")
        print(attn2.shape)

        # (batch_size, target_seq_len, d_model)

        attn2 = self.dropout2(attn2, training=training)
        print("DecoderLayer.call->attn2 after dropout")
        print(attn2.shape)

        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        print("DecoderLayer.call->out2")
        print(out2.shape)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        print("DecoderLayer.call->ffn_output")
        print(ffn_output.shape)

        ffn_output = self.dropout3(ffn_output, training=training)
        print("DecoderLayer.call->ffn_output after dropout")
        print(ffn_output.shape)

        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        print("DecoderLayer.call->out3")
        print(out3.shape)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        print("Encoder.call->x")
        print(x.shape)
        if mask is not None:
            print("Encoder.call->mask")
            print(mask.shape)

        seq_len = tf.shape(x)[1]
        print("Encoder.call->seq_len")
        print(seq_len)
        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        print("Encoder.call->x after embedding")
        print(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        print("Encoder.call->x after sqrt")
        print(x)
        x += self.pos_encoding[:, :seq_len, :]
        print("Encoder.call->x after pos encoding")
        print(x)
        x = self.dropout(x, training=training)
        print("Encoder.call->x after dropout")
        print(x)
        for i in range(self.num_layers):
            print("Encoder.call->start loop layers " + str(i))
            print("*****************************************")
            x = self.enc_layers[i](x, training, mask)
            print("Encoder.call->end loop layers " + str(i))
            print("*****************************************")
        print("Encoder.call->x after num_layers loop")
        print(x)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        print("Decoder.call->x")
        print(x.shape)
        if look_ahead_mask is not None:
            print("Decoder.call->look_ahead_mask")
            print(look_ahead_mask.shape)
        if padding_mask is not None:
            print("Decoder.call->padding_mask")
            print(padding_mask.shape)

        seq_len = tf.shape(x)[1]
        print("Decoder.call->seq_len")
        print(seq_len)

        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        print("Decoder.call->x after embedding")
        print(x.shape)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        print("Decoder.call->x after sqrt")
        print(x.shape)

        x += self.pos_encoding[:, :seq_len, :]
        print("Decoder.call->x after pos_encoding")
        print(x.shape)

        x = self.dropout(x, training=training)
        print("Decoder.call->x after dropout")
        print(x.shape)

        for i in range(self.num_layers):
            print("Decoder.call->start loop layers " + str(i))
            print("*****************************************")
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
            print("*****************************************")
            print("Decoder.call->end loop layers " + str(i))
        # x.shape == (batch_size, target_seq_len, d_model)
        print("Decoder.call->x after num_layers loop")
        print(x.shape)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, output_size, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(output_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        print("Transformer.call->enc_output")
        print(enc_output.shape)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        print("Transformer.call->dec_output")
        print(dec_output.shape)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        print("Transformer.call->final_output")
        print(final_output.shape)
        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(p_d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(p_d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    print("loss_function->real")
    print(real)
    print("loss_function->pred")
    print(pred)
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    # print("loss_function->mask")
    # print(mask)
    loss_ = loss_object(real, pred)
    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask
    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')

transformer = Transformer(num_layers=p_num_layers,
                          d_model=p_d_model,
                          num_heads=p_num_heads,
                          dff=p_dff,
                          output_size=p_output_size,
                          pe_input=p_time_steps,
                          pe_target=p_time_steps,
                          rate=p_dropout_rate)


def create_masks(inp, tar):
    print("create_masks->inp")
    print(inp.shape)
    print("create_masks->tar")
    print(tar.shape)

    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)
    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)
    print("create_masks->enc_padding_mask")
    print(enc_padding_mask.shape)
    print("create_masks->dec_padding_mask")
    print(dec_padding_mask.shape)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    print("create_masks->combined_mask")
    print(combined_mask.shape)
    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64), ]


# @tf.function(input_signature=train_step_signature)
@tf.function
def train_step(inp, tar):
    print("train_step->inp")
    print(inp.shape)
    print("train_step->tar")
    print(tar.shape)

    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]
    # print("train_step->tar_inp")
    # print(tar_inp.shape)
    # print("train_step->tar_real")
    # print(tar_real.shape)

    # TODO: enc_padding_mask, combined_mask, dec_padding_mask is now set to None
    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
    enc_padding_mask = combined_mask = dec_padding_mask = None

    if enc_padding_mask is not None:
        print("train_step->enc_padding_mask")
        print(enc_padding_mask.shape)
    if combined_mask is not None:
        print("train_step->combined_mask")
        print(combined_mask.shape)
    if dec_padding_mask is not None:
        print("train_step->dec_padding_mask")
        print(dec_padding_mask.shape)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar, True, enc_padding_mask, combined_mask, dec_padding_mask)
        print("train_step->predictions")
        print(predictions.shape)
        print(predictions)
        loss = loss_function(tar, predictions)
        # loss = loss_object(tar, predictions)
        print("train_step->tar")
        print(tar)

    print("tran_step->loss")
    print(loss.shape)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar, predictions)


for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp[p_input_names[0]], tar)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(),
                                                                         train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
