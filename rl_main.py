from feed.bt_data import BTCSVBasicData
from rl.agent.base import *
# from rl.agent.dqn_agent import DQNAgent
from rl.agent.dqn_agent_c import DQNAgent, DQNConfig
from rl.env.cerebro_ext import RLExtCerebro
import backtrader as bt
import tensorflow as tf

episode = 50
data_path = "./test_data/stock/000002.csv"
scaled_data_path = ""

network_config_path = "./config/stock_mlp_baseline.cls"


if __name__ == "__main__":
    with tf.Session() as sess:
        # agent = DQNAgent(network_config_path, sess)
        config = DQNConfig()
        agent = DQNAgent(config, sess)
        global_step = 0

        for i in range(episode):
            print("#####################EPISODE " + str(i) + "###########################")

            cerebro = RLExtCerebro()

            # Add a agent
            cerebro.addagent(agent)

            # Add a strategy
            cerebro.addstrategy(RLCommonStrategy)

            print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

            # Create a Data Feed
            data = BTCSVBasicData(
                dataname=data_path,
                reverse=False)

            # Add the Data Feed to Cerebro
            cerebro.adddata(data)

            # Add additional data frame
            cerebro.adddf(data_path, columns=RLExtCerebro.BASIC_COLUMNS)
            cerebro.addglobalstep(global_step)

            # Add sharpe ratio analyzer to Cerebro
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sp")
            # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

            # Add a FixedSize sizer according to the stake
            cerebro.addsizer(bt.sizers.FixedSize, stake=100)

            # Set the commission - 0.1% ... divide by 100 to remove the %
            cerebro.broker.setcommission(commission=0.001)

            # Set our desired cash start
            cerebro.broker.setcash(100000.0)

            cerebro.run()

            print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
            # cerebro.plot()

            global_step = cerebro.getglobalstep()
            # print(global_step)
