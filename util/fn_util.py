import tensorflow as tf

def get_act_fn(fn_name):
    if fn_name == 'tf.nn.relu':
        return tf.nn.relu

    if fn_name == 'tf.nn.relu6':
        return tf.nn.relu6

    raise FunctionNotFoundException()


def get_loss_fn(fn_name, y_, y):
    # y  is the prediction
    # y_ is the input label
    if fn_name == 'cross_entropy':
        # y should not be scaled
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    if fn_name == 'mean_squared_error':
        return tf.losses.mean_squared_error(y_, y)

    if fn_name == 'root_mean_squared_error':
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y))))

    if fn_name == 'r2_score':
        return None

    raise FunctionNotFoundException()


def get_opt_fn(fn_name):
    if fn_name == 'tf.train.GradientDescentOptimizer':
        return tf.train.GradientDescentOptimizer

    if fn_name == 'tf.train.AdamOptimizer':
        return tf.train.AdamOptimizer

    if fn_name == 'tf.train.AdagradDAOptimizer':
        return tf.train.AdagradDAOptimizer

    if fn_name == 'tf.train.AdadeltaOptimizer':
        return tf.train.AdadeltaOptimizer

    if fn_name == 'tf.train.ProximalAdagradOptimizer':
        return tf.train.ProximalAdagradOptimizer

    raise FunctionNotFoundException()


def get_acc_fn(fn_name, y_, y):
    with tf.name_scope(fn_name):
        if fn_name == 'correct_prediction':
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if fn_name == 'mean_squared_error':
            with tf.name_scope('accuracy'):
                return tf.losses.mean_squared_error(y_, y)

        if fn_name == 'root_mean_squared_error':
            with tf.name_scope('accuracy'):
                return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y))))

    raise FunctionNotFoundException()


class FunctionNotFoundException(Exception):
    def __init__(self):
        print("Function is not found")