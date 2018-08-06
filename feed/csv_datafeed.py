import numpy as np
import random
from sklearn.model_selection import train_test_split


@DeprecationWarning
class CSVDataFeed:
    def __init__(self, filenames):
        self.filenames = filenames
        print("Incoming files")
        print(filenames)
        self.train_filenames, self.test_filenames = self.__init_files()
        self.curr_data_pool = self.__load_csv(self.train_filenames[0])

        self.curr_train_file_idx = 0
        self.curr_test_file_idx = 0
        self.curr_batch_idx = 0
        self.offset = 0

    def __init_files(self):
        train_filenames, test_filenames = train_test_split(self.filenames, train_size=0.8)

        if len(train_filenames) == 0:
            train_filenames = test_filenames

        print("Files for training:")
        print(train_filenames)
        print("Files for testing")
        print(test_filenames)

        return train_filenames, test_filenames

    def __load_csv(self, filename):
        csv_data = np.loadtxt(fname=filename,
                              delimiter=',',
                              skiprows=1,
                              converters={0: lambda s: float(str.replace(s, '-', ''))}
                              )
        return csv_data

    def __dense_to_one_hot(self, x):
        arr = np.zeros(self.offset)
        try:
            idx = int(round(x) + 10)

            if idx >= self.offset - 1:
                idx = self.offset - 1
            elif idx < 0:
                idx = 0

            arr[idx] = 1
        except Exception as ex:
            print(ex.message)
            print(x)
        finally:
            return arr

    def get_next_train_batch_rnn(self, batch_size, one_hot=True, offset=21, raise_exception=False):
        # print (self.train_filenames[self.curr_train_file_idx])
        self.offset = offset
        while self.curr_batch_idx + batch_size > len(self.curr_data_pool):
            self.curr_batch_idx = 0
            if self.train_filenames[0] == self.test_filenames[0]:
                self.curr_batch_idx = 0
                if raise_exception:
                    raise EndOfBatchInterrupt("Only one file in dir, and end of the file")
            else:
                self.curr_train_file_idx = self.curr_train_file_idx + 1
                if self.curr_train_file_idx == len(self.train_filenames) - 1:
                    self.curr_train_file_idx = 0
                    if raise_exception:
                        raise EndOfBatchInterrupt("Already move to last file, reset to the first file")
                self.curr_data_pool = self.__load_csv(self.train_filenames[self.curr_train_file_idx])
                # print("Switch to next file")

        start_batch_idx = self.curr_batch_idx
        end_batch_idx = self.curr_batch_idx + batch_size
        self.curr_batch_idx = end_batch_idx

        train_data = self.curr_data_pool[start_batch_idx:end_batch_idx, ]
        batch_x = train_data[:, 1:-1]

        if one_hot:
            batch_y = np.array(map(self.__dense_to_one_hot, train_data[:, -1]))
        else:
            batch_y = train_data[:, -1]

        batch_y = np.reshape(batch_y, (batch_x.shape[0], batch_y.size / batch_x.shape[0]))
        # print(batch_x.shape)
        # print(batch_y.shape)
        # print("start idx:" + str(start_batch_idx) + " end idx:" + str(end_batch_idx)
        # + " len:" + str(len(self.curr_data_pool)))
        return batch_x, batch_y

    def get_next_train_batch(self, batch_size, one_hot=True, offset=21, raise_error=True):
        # print (self.train_filenames[self.curr_train_file_idx])
        self.offset = offset
        # print("Current batch idx:" + str(self.curr_batch_idx) + " data size:" + str(len(self.curr_data_pool)))
        if self.curr_batch_idx >= len(self.curr_data_pool) - 1:
            self.curr_batch_idx = 0
            if self.train_filenames[0] == self.test_filenames[0]:
                self.curr_batch_idx = 0
                if raise_error:
                    raise EndOfBatchInterrupt("Only one file in dir, and end of the file")
            else:
                self.curr_train_file_idx = self.curr_train_file_idx + 1
                if self.curr_train_file_idx == len(self.train_filenames) - 1:
                    self.curr_train_file_idx = 0
                    if raise_error:
                        raise EndOfBatchInterrupt("Already move to last file, reset to the first file")
                self.curr_data_pool = self.__load_csv(self.train_filenames[self.curr_train_file_idx])
                # print("Switch to next file")

        start_batch_idx = self.curr_batch_idx
        end_batch_idx = self.curr_batch_idx + batch_size
        self.curr_batch_idx = end_batch_idx

        train_data = self.curr_data_pool[start_batch_idx:end_batch_idx, ]
        batch_x = train_data[:, 1:-1]

        if one_hot:
            batch_y = np.array(map(self.__dense_to_one_hot, train_data[:, -1]))
        else:
            batch_y = train_data[:, -1]

        # batch_y = np.reshape(batch_y, (batch_x.shape[0], batch_y.size / batch_x.shape[0]))
        # print(batch_x.shape)
        # print(batch_y.shape)
        # print("start idx:" + str(start_batch_idx) + " end idx:" + str(end_batch_idx)
        # + " len:" + str(len(self.curr_data_pool)))
        return batch_x, batch_y
        # set_file = False
        # start_batch_idx = self.curr_batch_idx
        # end_batch_idx = self.curr_batch_idx + batch_size
        # if end_batch_idx > len(self.curr_data_pool):
        #     end_batch_idx = len(self.curr_data_pool)
        #     set_file = True
        # self.curr_batch_idx = end_batch_idx
        #
        # # print("Current start idx: " + str(start_batch_idx) + " and end idx: " + str(end_batch_idx))
        # train_data = self.curr_data_pool[start_batch_idx:end_batch_idx, ]
        # batch_x = train_data[:, 1:-1]
        #
        # if one_hot:
        #     batch_y = np.array(map(self.__dense_to_one_hot, train_data[:, -1]))
        # else:
        #     batch_y = train_data[:, -1]
        #
        # if set_file:
        #     self.curr_batch_idx = 0
        #     if self.train_filenames [0] == self.test_filenames[0]:
        #         self.curr_batch_idx = 0
        #         raise EndOfBatchInterrupt("Only one file in dir, and end of the file")
        #     else:
        #         self.curr_batch_idx = 0
        #         self.curr_train_file_idx = self.curr_train_file_idx + 1
        #         if self.curr_train_file_idx == len(self.train_filenames) - 1:
        #             self.curr_train_file_idx = 0
        #             raise EndOfBatchInterrupt("Already move to last file, reset to the first file")
        #         self.curr_data_pool = self.__load_csv(self.train_filenames[self.curr_train_file_idx])
        #         # print("Switch to next file")
        #     set_file = False
        # # batch_y = np.reshape(batch_y, (batch_x.shape[0], batch_y.size/batch_x.shape[0]))
        # return batch_x, batch_y

    def get_test_batch(self, batch_size=None, one_hot=True, offset=21):
        def get_test_data():
            idx = random.randint(0, len(self.test_filenames) - 1)
            # print("Current file idx:" + str(idx))
            return self.__load_csv(self.test_filenames[idx])

        self.offset = offset
        test_data = get_test_data()

        while test_data.shape[0] < batch_size:
            test_data = get_test_data()

        if batch_size is None:
            batch_x = test_data[:, 1:-1]
            if one_hot:
                batch_y = np.array(map(self.__dense_to_one_hot, test_data[:, -1]))
            else:
                batch_y = test_data[:, -1]
        else:
            # print(test_data.shape)
            try:
                row_num = random.randint(0, test_data.shape[0] - 1 - batch_size)
            except Exception as _:
                row_num = 0
            # print("Current batch start idx:" + str(row_num))
            batch_x = test_data[row_num:row_num + batch_size, 1:-1]
            if one_hot:
                batch_y = np.array(map(self.__dense_to_one_hot, test_data[row_num:row_num + batch_size, -1]))
            else:
                batch_y = test_data[row_num:row_num + batch_size, -1]
        batch_y = np.reshape(batch_y, (batch_x.shape[0], batch_y.size / batch_x.shape[0]))
        return batch_x, batch_y


class EndOfBatchInterrupt(Exception):
    def __init__(self, msg):
        self.msg = msg
        # print(msg)
