import random
from os.path import join

from feed.bt_data import BTCSVBasicData
from rl.v2.agents.dqn import DQNAgent
from rl.v2.bt_ext.cerebro_ext import RLCerebro
from rl.v2.bt_ext.strategy_ext import RLCommonStrategy

data_path = "./test_data/stock/basic"
data_files = "./test_data/stock/basic/000002.csv"


def get_data_files(file_list):
    index = random.randint(0, len(file_list) - 1)
    csv_file = join(data_path, file_list[index])
    scaled_csv_file = join(data_path, file_list[index].replace(".csv", "_s.csv"))
    return csv_file, scaled_csv_file


if __name__ == "__main__":
    # data_files = [f for f in listdir(data_path) if f != ".DS_Store" and "_s" not in f]
    print(data_files)
    agent = DQNAgent()

    cerebro = RLCerebro()

    # Add a agent
    cerebro.addagent(agent)

    # Add a strategy
    cerebro.addstrategy(RLCommonStrategy)

    # Create a Data Feed
    data = BTCSVBasicData(
        dataname=data_files,
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.run()

    final_portfolio = round(cerebro.broker.getvalue(), 2)
    print("Final Portfolio Value: " + str(final_portfolio))
