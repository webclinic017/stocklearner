import numpy as np
import tensorflow as tf


def one_hot(source_slide, boundaries=[-7, -5, -3, 0, 3, 5, 7], on_value=1, off_value=0):
    # print(source_slide)
    def _one_hot(_source_value):
        # print(_source_value)
        _one_hot_value = [off_value for _ in range(len(boundaries) + 1)]
        if _source_value > boundaries[-1]:
            _one_hot_value[-1] = on_value
            return _one_hot_value
        for _idx, _ in enumerate(boundaries):
            if _source_value <= boundaries[_idx]:
                _one_hot_value[_idx] = on_value
                break
        return _one_hot_value

    if (type(source_slide)) is list or (type(source_slide) is np.ndarray):
        one_hot_slide = []
        for each in source_slide:
            one_hot_slide.append(_one_hot(each))
    else:
        one_hot_slide = _one_hot(source_slide)
    return one_hot_slide


def dict_to_list(dict):
    list = []
    for i in dict:
        list.append(dict[i])
    return np.stack(list, axis=1)

def rnn_output_split(list, time_steps, output_size):
    reshaped_narray = np.array(list).reshape(-1, time_steps, output_size)
    hsplited_narray = np.array(np.hsplit(reshaped_narray, time_steps)[-1])
    return hsplited_narray.reshape(-1, output_size)

def get_rnn_cells(cell_type, hidden_cells, forget_bias=1.0):
    if cell_type == "LSTM":
        return tf.contrib.rnn.LSTMCell(hidden_cells, forget_bias=forget_bias)

    if cell_type == "GRU":
        return tf.contrib.rnn.GRUCell(hidden_cells)

    if cell_type == "BasicLSTM":
        return tf.contrib.rnn.BasicLSTMCell(hidden_cells, forget_bias=forget_bias)

    raise CellTypeNotFoundException()


class CellTypeNotFoundException(Exception):
    def __init__(self):
        print("Function is not found")