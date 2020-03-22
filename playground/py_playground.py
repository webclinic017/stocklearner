def fn_t(a, *args, **kargs):
    print(a)
    print(args)
    print(kargs["b"])
    if "c" in kargs.keys():
        print(kargs["c"])

fn_t("a", b="b", c="c")

import tensorflow as tf
print(tf.__version__)

import numpy as np

# a = np.array(range(10))
# print(a)
# print(a[1::2])
# print(a[2::2])
# print(a[2::3])

position = 10
d_model = 512
d1 = np.arange(position)[:, np.newaxis]
d2 = np.arange(d_model)[np.newaxis, :]

print(d1.shape)
print(d2.shape)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


ag = get_angles(d1, d2, d_model)
print(ag.shape)

ll = np.arange(410).reshape(10, 41)
print(ll)
print(ll.shape)
ll2 = ll[:, :, np.newaxis]
print(ll2)
print(ll2.shape)
print("**********")

angle_rads = get_angles(ll2, np.arange(d_model)[np.newaxis, :], d_model)
print(angle_rads.shape)
# 将 sin 应用于数组中的偶数索引（indices）；2i
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
print(angle_rads.shape)
# 将 cos 应用于数组中的奇数索引；2i+1
angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
print(angle_rads.shape)
pos_encoding = angle_rads[np.newaxis, ...]
print(pos_encoding.shape)

# ll3 = np.ndarray(shape=(10, 41), buffer=np.arange(10))
n = np.array([np.append(np.arange(0), np.arange(41), axis=0) for i in range(10)])
print(n)
print(n.shape)
# print(ll3)

t1 = np.zeros((32, 1, 1, 1, 41))
print(t1.shape)
t2 = np.ones((32, 8, 10, 41, 41))
print(t2.shape)

# scaled_attention_logits += (mask * -1e9)
t2 += (t1 * -1e9)
print(t2.shape)
