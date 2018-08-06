import numpy as np


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