# import tensorflow as tf
# # Naive LSTM to learn three-char time steps to one-char mapping
# import numpy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
#
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # define the raw dataset
# alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#
# # create mapping of characters to integers (0-25) and the reverse
# char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# int_to_char = dict((i, c) for i, c in enumerate(alphabet))
#
# # prepare the dataset of input to output pairs encoded as integers
# seq_length = 3
# dataX = []
# dataY = []
#
# for i in range(0, len(alphabet) - seq_length, 1):
#     seq_in = alphabet[i:i + seq_length]
#     seq_out = alphabet[i + seq_length]
#     dataX.append([char_to_int[char] for char in seq_in])
#     dataY.append(char_to_int[seq_out])
#     print(seq_in, '->', seq_out)
#
# print(dataX)
# print(dataY)
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# print(X)
# # normalize
# X = X / float(len(alphabet))
# print(X.shape)
# # one hot encode the output variable
# print(X)
# y = tf.keras.utils.to_categorical(dataY)
# print(y)
# print(y.shape)
# # create and fit the model
# model = Sequential()
# model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, nb_epoch=500, batch_size=1, verbose=2)
#
# # summarize performance of the model
# scores = model.evaluate(X, y, verbose=0)
#
# print("Model Accuracy: %.2f%%" % (scores[1]*100))
# # demonstrate some model predictions
# for pattern in dataX:
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(len(alphabet))
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in pattern]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((10000, 20))
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)

x_test = np.random.random((100, 20))
y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

x_predict = np.random.random((1, 20))
y_predict = tf.keras.utils.to_categorical(np.random.randint(10, size=(1, 1)), num_classes=10)

print(x_predict)
print(y_predict)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train,
          epochs=500,
          batch_size=128,
          verbose=0)
score = model.evaluate(x_test, y_test, batch_size=128)
predict = model.predict(x_predict)
predict_c = model.predict_classes(x_predict)
print(predict)
print(np.argmax(predict))
print(predict_c)
