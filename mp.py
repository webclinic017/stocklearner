import multiprocessing
import time
import configparser
from os.path import join
from os import listdir
from util import model_util
from feed import csv_data as stock_data

CONFIG_FILE_PATH = "./app.config_file"


def func(msg, t):
    print(multiprocessing.current_process().name + '-' + msg)
    time.sleep(t)


def machine_learning(f, t, s):
    print(multiprocessing.current_process().name + "-" + f)
    # model = model_util.get_model(config_file)
    # model.train(train_dataset)
    # model.eval(eval_dataset)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    training_data_path = config.get("Data", "training_data_path")
    eval_data_path = config.get("Data", "eval_data_path")

    pool_size = config.getint("Execution", "pool_size")
    config_list_path = config.get("Execution", "config_list_path")

    config_file_list = [join(config_list_path, f) for f in listdir(config_list_path) if f != ".DS_Store"]
    print("Pool size: " + str(pool_size))
    print("Config file list path: " + config_list_path)

    train_dataset = stock_data.csv_input_fn(training_data_path)
    eval_dataset = stock_data.csv_input_fn(eval_data_path)

    pool = multiprocessing.Pool(processes=pool_size)  # 创建4个进程
    # for i in range(10):
    i = 1
    for config_file in config_file_list:
        # msg = "hello %d" %(i)
        print("Current config_file file is: " + config_file)
        # pool.apply_async(func, (config_file, i))
        pool.apply_async(machine_learning, (config_file, "", ""))
        i = i + 1
    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    print("Sub-process(es) done.")
