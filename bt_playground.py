from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
from feed.bt_data import BTCSVBasicData


# Create a Strategy
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)


    def start(self):
        self.log("Strategy fn_start, I should only see once")

    def prenext(self):
        self.log("Strategy fn_prenext")

    def next(self):
        self.log("Strategy fn_next")

    def nexstart(self):
        self.log("Strategy fn_nexstart")

    def stop(self):
        self.log("Strategy fn_end, I should only see at the end")

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    data_path = "./test_data/stock/000002.csv"

    # Create a Data Feed
    data = BTCSVBasicData(
        dataname=data_path,
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # cerebro.plot()
