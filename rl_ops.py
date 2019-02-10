from feed.bt_data import BTCSVBasicData
from rl.agent.base import *
from rl.agent.dqn_agent_c import DQNAgent, DQNConfig
from rl.env.cerebro_ext import RLExtCerebro
from rl.env.sizer_ext import PercentSizer
from os import listdir
from os.path import join
import tensorflow as tf
import random
import time

episode = 50
data_dir = "D:\\Output\\Train\\"
network_config_path = "./config_file/stock_mlp_baseline.cls"


def get_data_files(file_list):
    index = random.randint(0, len(file_list) - 1)
    csv_file = join(data_dir, file_list[index])
    scaled_csv_file = join(data_dir, file_list[index].replace(".csv", "_s.csv"))
    return csv_file, scaled_csv_file


if __name__ == "__main__":
    data_files = [f for f in listdir(data_dir) if f != ".DS_Store" and "_s" not in f]

    with tf.Session() as sess:
        config = DQNConfig()
        agent = DQNAgent(config, sess)
        global_step = 0

        for i in range(episode):
            print("#####################EPISODE " + str(i) + "###########################")

            data_path, scaled_data_path = get_data_files(data_files)
            print("Current file is: " + data_path)
            print("Related scaled file is: " + scaled_data_path)

            cerebro = RLExtCerebro()

            # Add a agent
            cerebro.addagent(agent)

            # Add a strategy
            cerebro.addstrategy(RLCommonStrategy)

            # Create a Data Feed
            data = BTCSVBasicData(
                dataname=data_path,
                reverse=False)

            # Add the Data Feed to Cerebro
            cerebro.adddata(data)

            # Add additional data frame
            cerebro.adddf(data_path, columns=RLExtCerebro.BASIC_COLUMNS)
            cerebro.addscaleddf(scaled_data_path, columns=RLExtCerebro.BASIC_COLUMNS)
            cerebro.addglobalstep(global_step)

            # Add sharpe ratio analyzer to Cerebro
            # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

            # Add a FixedSize sizer according to the stake
            # cerebro.addsizer(bt.sizers.FixedSize, stake=100)
            cerebro.addsizer(PercentSizer)

            # Set the commission - 0.1% ... divide by 100 to remove the %
            cerebro.broker.setcommission(commission=0.001)

            # Set our desired cash start
            cerebro.broker.setcash(100000.0)
            print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

            cerebro.run()

            final_portfolio = round(cerebro.broker.getvalue(), 2)
            print("Final Portfolio Value: " + str(final_portfolio))
            # thestrat = thestrats[0]
            # print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
            assert final_portfolio >= 0, "????"

            # cerebro.plot()
            global_step = cerebro.getglobalstep()
            time.sleep(5)
            # print(global_step)
