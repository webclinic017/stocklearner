import numpy as np
import pandas as pd


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

# import gym
# env = gym.make('MountainCar-v0')
# for i_episode in range(200):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         # print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

# t_np = np.array([True]) + 0.
# print(t_np)

#              0       1       2       3        4      5         6               7           8           9
BASIC_COLUMNS = ["DATE", "OPEN", "HIGH", "CLOSE", "LOW", "VOLUME", "PRICE_CHANGE", "P_CHANGE", "TURNOVER", "LABEL"]

df = pd.read_csv("./test_data/stock/000002.csv")
df.columns = BASIC_COLUMNS
df.set_index("DATE", inplace=True)
if "LABEL" in BASIC_COLUMNS:
    df.drop(["LABEL"], axis=1, inplace=True)
# print(df.loc["2016-12-13"])
date = "1991-06-13"
idx = df.index.get_loc(date)
offset = -5
print(idx)
print(idx - offset + 1)
if offset != 0:
    offset = offset + 1
    df2 = df.iloc[idx + offset :idx + 1]
else:
    df2 = df.loc[date]
print(df2)
print(df2.values.tolist())
