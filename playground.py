import numpy as np

'''
OUTPUT_SIZE = 2
TIME_STEPS = 2

a = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8]])
print(a)
print(a.shape)
print("-----------------------")
b = np.reshape(a, (-1, TIME_STEPS, OUTPUT_SIZE))
print(b)
print(b.shape)
print("-----------------------")
c = np.array(np.hsplit(b, 2)[-1])
print(c)
print(c.shape)
print(c.reshape(-1, OUTPUT_SIZE))
print("-----------------------")

def transform_rnn_output(list, time_steps, output_size):
    reshaped_narray = np.array(list).reshape(-1, time_steps, output_size)
    hsplited_narray = np.array(np.hsplit(reshaped_narray, time_steps)[-1])
    return hsplited_narray.reshape(-1, output_size)

lst = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9]]
ary = a(lst, 3, 2)
print(ary)
'''

# import logging
#
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
#
# handler = logging.FileHandler("test_log.txt")
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
#
# logger.addHandler(handler)
#
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")


class Parent:
    def __init__(self, name):
        print(name)
        self.name = name
        self.__fn_a()
        self.__fn_b()

    def __fn_a(self):
        print("Parent->fn_a")

    def __fn_b(self):
        print("Parent->fn_b")

    def fn_c(self):
        print("Parent->fn_c")


class Child(Parent):
    def __init__(self, name):
        Parent.__init__(self, name)
        print(name)
        self.__fn_a()
        self.__fn_b()

    def __fn_a(self):
        print("Child->fn_a")

    def __fn_b(self):
        print("Child->fn_b")
        self.fn_c()

some = Parent("1")
print("------------------")
another = Child("2")
print("++++++++++++++++++")
print(another.name)
# another.fn_c()


def test(func):
    func()
    print
    "call test"


def test1(f):
    f()
    print
    "call test1"


def main():
    @test
    def fun(a, b):
        print
        "call fun"

        @test1
        def fun1():
            print
            "call fun1"
